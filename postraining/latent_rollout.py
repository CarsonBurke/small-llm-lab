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
caller (the DAPO trainer writes exact verifier reward plus bounded numeric
distance shaping through ``assign_terminal_rewards``);
``continuation_reward`` survives only for the ``sample_latent --fineweb``
inspection tool.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, fields

import torch
from torch import Tensor

from postraining.core import top_p_sample
from postraining.latent_thought import EMIT, THINK, LatentThoughtModel, StepOutput

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
    # (batch, stream, dim): old diagonal-Gaussian factors retained so replay
    # can clip every latent dimension against the frozen behavior policy.
    old_thought_logprobs: Tensor
    # (batch, stream, dim), fp32: the behavior policy's Gaussian parameters at
    # THINK positions. The projected THINK objective measures its Mahalanobis
    # trust region against these directly instead of estimating divergence
    # from single-sample ratios. Filled by refresh_old_statistics through the
    # exact replay path, like the log-probabilities above.
    old_thought_means: Tensor
    old_thought_log_sigmas: Tensor
    old_values: Tensor
    rewards: Tensor
    reward_scalar: Tensor  # (batch,)
    prompt_length: int
    # Set by refresh_old_statistics: old_values/old_*_logprobs have been
    # recomputed through the update-step replay path. The actor update
    # refuses unrefreshed batches; a tensor-width proxy cannot express this
    # for pinned-EMIT rollouts, whose thought tensors are always zero-width.
    statistics_refreshed: bool = False

    def to(
        self, device: torch.device, non_blocking: bool = False
    ) -> "LatentRolloutBatch":
        moved = {}
        for field in fields(self):
            value = getattr(self, field.name)
            moved[field.name] = (
                value.to(device, non_blocking=non_blocking)
                if isinstance(value, Tensor)
                else value
            )
        return LatentRolloutBatch(**moved)

    @property
    def stream_length(self) -> int:
        return self.kind.size(1)


def compact_stream_to_device(
    batch: LatentRolloutBatch, device: torch.device
) -> LatentRolloutBatch:
    """Copy only the used stream prefix into owning storage on ``device``.

    Rollout storage is provisioned for the full context, while ordinary
    trajectories use only a small prefix. Slicing before a device transfer
    avoids copying the unused tail and the destination copy already owns its
    compact storage, unlike a same-device narrow view.
    """
    used = int((batch.kind != PAD_SLOT).any(0).nonzero().max()) + 1
    compact = {}
    async_cuda_to_cpu = (
        batch.kind.device.type == "cuda" and device.type == "cpu"
    )
    for field in fields(batch):
        value = getattr(batch, field.name)
        if (
            isinstance(value, Tensor)
            and value.dim() >= 2
            and value.size(1) == batch.stream_length
        ):
            value = value[:, :used]
        if isinstance(value, Tensor):
            if async_cuda_to_cpu:
                target = torch.empty_like(
                    value, device=device, pin_memory=True
                )
                target.copy_(value, non_blocking=True)
                value = target
            elif value.device == device:
                value = value.clone()
            else:
                value = value.to(device)
        compact[field.name] = value
    if async_cuda_to_cpu:
        # All pinned copies are enqueued on the current stream. One barrier
        # makes the complete compact batch host-readable without serializing
        # once per dataclass field.
        torch.cuda.current_stream(batch.kind.device).synchronize()
    return LatentRolloutBatch(**compact)


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
    prompt_repeats: int = 1,
    pin_emit: bool = False,
    tail_caches: list[tuple[Tensor, ...]] | None = None,
    tail_step_core=None,
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

    ``pin_emit`` fixes every gate decision to EMIT without sampling it: the
    rollout is a plain token policy (cot/none reasoning modes). No gate or
    thought RNG is consumed, ``gate_mask`` stays all-zero (no gate action was
    ever taken, so every gate/thought loss term degrades to zero through its
    mask), and thought storage keeps a zero-width final dimension exactly
    like ``replay_storage=False``.

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

    ``prompt_repeats`` declares how many output trajectories each UNIQUE input
    prompt owns. Its deterministic prefix is evaluated once, then its final
    policy output, prompt storage, and populated KV prefix are expanded across
    members before stochastic actions consume RNG. This structural interface
    makes accidentally supplying unequal repeated prompts impossible.

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

    ``tail_caches`` (with ``finished_batch_size``) switches the compacted tail
    onto caller-owned static caches (``make_static_generation_cache`` of
    exactly ``finished_batch_size`` rows, at least ``max_stream`` long):
    at the tail compaction the survivors' live cache prefix is copied into
    them and every remaining step attends the full static width under a
    fixed-shape per-row ``key_mask`` that combines causality with each row's
    left-pad validity — constant shapes, so ``tail_step_core`` (a
    CUDA-graph-compiled ``step_core``) replays one graph per step. Reusing one
    cache set across rollouts of identical shape keeps the graph from
    re-recording; stale values it carries stay masked, exactly like the zero
    fill at allocation. Without ``tail_step_core`` the tail runs through the
    ordinary ``wrapper.step`` (the eager 2-D-mask path), which is what CPU
    tests exercise.
    """
    if prompt_ids.dim() != 2 or prompt_ids.size(1) < 1:
        raise ValueError("prompt_ids must be (batch, length>=1)")
    if max_stream_steps < max_new_tokens:
        raise ValueError("max_stream_steps must be at least max_new_tokens")
    device = prompt_ids.device
    prefix_batch, prompt_length = prompt_ids.shape
    if prompt_repeats < 1:
        raise ValueError("prompt_repeats must be positive")
    batch = prefix_batch * prompt_repeats
    if caches is not None and prompt_repeats != 1:
        raise ValueError("prompt_repeats cannot be used with preallocated caches")
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
    if pin_emit and bool(force_initial_think.any()):
        raise ValueError(
            "pin_emit rollouts cannot force an initial THINK: the gate is "
            "never sampled and no thought machinery runs"
        )
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
        if prompt_lengths.shape != (prefix_batch,):
            raise ValueError("prompt_lengths must be one true length per prompt")
        prompt_lengths = prompt_lengths.to(device=device, dtype=torch.long)
        if bool((prompt_lengths < 1).any()) or bool(
            (prompt_lengths > prompt_length).any()
        ):
            raise ValueError("prompt_lengths must lie in [1, prompt_ids width]")
        pad_lengths = (
            prompt_length - prompt_lengths
        ).repeat_interleave(prompt_repeats)
        # Slot k is a real (attendable) slot for row b iff k >= pad_lengths[b].
        valid_slots = (
            torch.arange(max_stream, device=device)[None, :] >= pad_lengths[:, None]
        )
    preallocated_caches = caches is not None
    if caches is None:
        cache_batch = prefix_batch if prompt_repeats > 1 else batch
        cache_length = prompt_length if prompt_repeats > 1 else max_stream
        caches = wrapper.make_generation_cache(
            cache_batch, cache_length, device, dtype=cache_dtype
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

    tail_mask: Tensor | None = None
    tail_length = 0
    if tail_caches is not None:
        if preallocated_caches:
            raise ValueError(
                "tail_caches is the mid-rollout static switch; preallocated "
                "caches are already static for the whole rollout"
            )
        if finished_batch_size is None or not compact_finished:
            raise ValueError(
                "tail_caches requires compact_finished and "
                "finished_batch_size: the static switch happens at the "
                "fixed-size tail compaction"
            )
        if position_index is None:
            raise ValueError("tail_caches requires tensor_positions")
        tail_length = tail_caches[0][0].size(2)
        if (
            tail_caches[0][0].size(0) != finished_batch_size
            or tail_length < max_stream
        ):
            raise ValueError(
                f"tail caches ({tuple(tail_caches[0][0].shape)}) do not fit "
                f"tail batch {finished_batch_size} x stream {max_stream}"
            )
        if tail_caches[0][0].dtype != caches[0][0].dtype:
            raise ValueError(
                f"tail cache dtype {tail_caches[0][0].dtype} must match the "
                f"rollout cache dtype {caches[0][0].dtype}"
            )

    def step_position(position: int) -> tuple[int | Tensor, Tensor | None]:
        if tail_mask is not None:
            # Fixed-shape tail: the full-width row mask grows by one column
            # per step in place, so the step's shapes never change and the
            # mask write is one tiny kernel. Generated slots are always
            # attendable; only each row's left-pad prefix stays False.
            position_index.fill_(position)
            tail_mask[:, position] = True
            return position_index, tail_mask
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
    stored_thought_dim = model_dim if replay_storage and not pin_emit else 0
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
    old_thought_means = thoughts.new_zeros((batch, max_stream, 0))
    old_thought_log_sigmas = thoughts.new_zeros((batch, max_stream, 0))
    old_values = torch.zeros_like(action_mask)

    if valid_slots is None:
        kind[:, :prompt_length] = TOKEN_SLOT
        token_ids[:, :prompt_length] = prompt_ids.repeat_interleave(
            prompt_repeats, dim=0
        )
    else:
        prompt_valid = valid_slots[:, :prompt_length]
        kind[:, :prompt_length] = torch.where(prompt_valid, TOKEN_SLOT, PAD_SLOT)
        token_ids[:, :prompt_length] = (
            prompt_ids.repeat_interleave(prompt_repeats, dim=0)
            * prompt_valid
        )

    prefix_prompt_ids = prompt_ids
    prefix_valid_slots = (
        valid_slots[::prompt_repeats] if valid_slots is not None else None
    )
    prefix_key_valid = (
        prefix_valid_slots[:, :prompt_length]
        if prefix_valid_slots is not None
        else None
    )
    output = wrapper.prefill(prefix_prompt_ids, caches, prefix_key_valid)
    caches = output.caches

    if prompt_repeats > 1:
        expanded_caches = wrapper.make_generation_cache(
            batch, max_stream, device, dtype=cache_dtype
        )
        for source_layer, target_layer in zip(
            caches, expanded_caches, strict=True
        ):
            for source, target in zip(source_layer, target_layer, strict=True):
                grouped_target = target.view(
                    prefix_batch,
                    prompt_repeats,
                    *target.shape[1:],
                )
                grouped_target[:, :, :, :prompt_length].copy_(
                    source[:, None].expand(
                        prefix_batch,
                        prompt_repeats,
                        *source.shape[1:],
                    )
                )
        caches = expanded_caches

        def expand_rows(value: Tensor) -> Tensor:
            return value.repeat_interleave(prompt_repeats, dim=0)

        output = output.__class__(
            belief=expand_rows(output.belief),
            predicted=expand_rows(output.predicted),
            thought_log_sigma=expand_rows(output.thought_log_sigma),
            input_latent=expand_rows(output.input_latent),
            logits=expand_rows(output.logits),
            caches=caches,
        )

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
            snap_to_tail = (
                finished_batch_size is not None
                and active_count <= finished_batch_size
            )
            if snap_to_tail:
                compacted_count = min(current_count, finished_batch_size)
            # Above the tail width, compact progressively under the same
            # >=25%-dead hysteresis as the tail-free path: a few long
            # survivors must not keep stepping hundreds of finished rows at
            # full width until the final tail snap. Intermediate widths never
            # equal the static tail size, so they run on the dynamic-shape
            # step core and cannot engage the tail graph early.
            should_compact = (
                compact_finished
                and not preallocated_caches
                and compacted_count < current_count
                and (
                    snap_to_tail
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
                if (
                    tail_caches is not None
                    and compacted_count == tail_caches[0][0].size(0)
                ):
                    # Static-tail switch: land the survivors' live prefix in
                    # the caller-owned graph-static caches. Slots at or after
                    # ``live_prefix`` keep stale-but-finite values from
                    # earlier rollouts; the fixed-width row mask hides them,
                    # exactly like the zero fill at allocation.
                    for layer, cache in enumerate(caches):
                        static_layer = tail_caches[layer]
                        for tensor, target in zip(
                            cache, static_layer, strict=True
                        ):
                            target[:, :, :live_prefix].copy_(
                                tensor[:, :, :live_prefix].index_select(
                                    0, keep
                                )
                            )
                        # Replace one layer at a time so the dynamic caches
                        # free as the static ones fill.
                        caches[layer] = static_layer
                    tail_mask = torch.zeros(
                        (compacted_count, tail_length),
                        dtype=torch.bool,
                        device=device,
                    )
                    if valid_slots is not None:
                        # Rows were compacted above, so this is the
                        # survivors' causal-and-valid mask at the switch.
                        tail_mask[:, :live_prefix] = valid_slots[
                            :, :live_prefix
                        ]
                    else:
                        tail_mask[:, :live_prefix] = True
                elif snap_to_tail or finished_batch_size is None:
                    for layer, cache in enumerate(caches):
                        compacted = []
                        for tensor in cache:
                            target = torch.empty(
                                (compacted_count, *tensor.shape[1:]),
                                dtype=tensor.dtype,
                                device=tensor.device,
                            )
                            target[:, :, :live_prefix].copy_(
                                tensor[:, :, :live_prefix].index_select(
                                    0, keep
                                )
                            )
                            compacted.append(target)
                        # Replace one layer at a time so old+new
                        # full-capacity caches do not coexist across all
                        # layers at peak memory.
                        caches[layer] = tuple(compacted)
                else:
                    # Progressive above-tail compaction gathers the
                    # survivors into the FRONT of the existing storage and
                    # keeps narrowed contiguous views. Reallocating here
                    # (the branch above) adds a full-capacity transient per
                    # tensor at peak KV pressure — that exact allocation
                    # OOMed the v23 smoke run. The view keeps the original
                    # storage alive until the tail snap frees it, which is
                    # the same footprint the pre-progressive code held at
                    # full width; the transient shrinks to survivors x
                    # live-prefix. The survivor gather is materialized
                    # BEFORE the in-place copy, so overlapping rows cannot
                    # alias.
                    for layer, cache in enumerate(caches):
                        compacted = []
                        for tensor in cache:
                            survivors = tensor[
                                :, :, :live_prefix
                            ].index_select(0, keep)
                            tensor[
                                :compacted_count, :, :live_prefix
                            ].copy_(survivors)
                            compacted.append(tensor[:compacted_count])
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
        if pin_emit:
            # No gate action exists in this policy: nothing is sampled and
            # gate_mask/old_gate_logprobs stay zero everywhere.
            action = torch.full(
                (belief.size(0),), EMIT, dtype=torch.long, device=device
            )
            gate_logprob = None
        elif record_likelihoods:
            action, gate_logprob = wrapper.gate.sample(
                belief, generator=generator
            )
        else:
            action = wrapper.gate.sample_action(belief, generator=generator)
            gate_logprob = None
        if initial and not pin_emit:
            action = action.masked_fill(force_initial_think, THINK)

        # RNG draw order (gate, token, thought) is part of the execution
        # schema; pin_emit skips the gate/thought draws entirely but must not
        # reorder the latent path's consumption.
        token = top_p_sample(output.logits, temperature, top_p)
        token_logprob = None
        if record_likelihoods:
            token_logprob = (
                output.logits.float()
                .log_softmax(-1)
                .gather(-1, token[:, None])
                .squeeze(-1)
            )
        thought = (
            None
            if pin_emit
            else wrapper.transition.sample_latent(
                output.predicted,
                output.thought_log_sigma,
                generator=generator,
            )
        )

        record = active
        record_rows = live_rows[record]
        action_mask[record_rows, position] = 1.0
        if not pin_emit:
            sampled_gate = record & (~force_initial_think if initial else True)
            sampled_gate_rows = live_rows[sampled_gate]
            gate_mask[sampled_gate_rows, position] = 1.0
            if gate_logprob is not None:
                old_gate_logprobs[sampled_gate_rows, position] = gate_logprob[
                    sampled_gate
                ].float()
        gate_actions[record_rows, position] = action[record]
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
        if replay_storage and thought is not None:
            thoughts[think_rows, next_position] = thought[thinks]
        emitted += emits.long()
        if stop_tensor is not None:
            ended |= emits & torch.isin(token, stop_tensor)

        # Finished rows keep stepping on token 0 (their next slot stays PAD,
        # so the zero-initialized token_ids row feeds the embedding; every
        # loss masks it); batched caches make per-row early exit impractical.
        next_kinds = kind[live_rows, next_position]
        next_token_ids = token_ids[live_rows, next_position]
        if thought is None:
            next_input = wrapper.embed_tokens(next_token_ids[:, None])
        else:
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
        if tail_mask is not None and tail_step_core is not None:
            belief, predicted, thought_log_sigma, logits = tail_step_core(
                next_input, caches, step_pos, key_mask
            )
            output = StepOutput(
                belief=belief,
                predicted=predicted,
                thought_log_sigma=thought_log_sigma,
                input_latent=next_input.squeeze(1),
                logits=logits,
                caches=caches,
            )
        else:
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
        old_thought_means=old_thought_means,
        old_thought_log_sigmas=old_thought_log_sigmas,
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


def pack_rollout_groups_for_replay(
    groups: Sequence[LatentRolloutBatch],
    pin_memory: bool = False,
) -> LatentRolloutBatch:
    """Right-pad independent prompt groups into one exact replay batch.

    ``pin_memory`` allocates the packed CPU tensors in page-locked memory
    (via the caching host allocator, so repeated same-size packs recycle
    their blocks) so the subsequent H2D upload can run as a fast DMA copy;
    it is ignored for on-device groups or CUDA-less hosts.

    Every source stream keeps its original position zero and action boundary;
    only unused tail columns are appended. This matters because full-sequence
    replay is causal but has no left-padding key mask. A conventional
    left-padded batch would therefore change attention normalization and break
    behavior-policy log-probability equality.

    The result is intended for replay/update, where action masks carry the
    per-row boundaries. Its scalar ``prompt_length`` is the minimum source
    prompt length solely so length-aware row selection remains valid; callers
    must decode/score the individual groups before combining them.
    """
    if not groups:
        raise ValueError("at least one rollout group is required")
    devices = {group.kind.device for group in groups}
    if len(devices) != 1:
        raise ValueError("rollout groups must share one device")
    max_stream = max(group.stream_length for group in groups)
    total_rows = sum(group.kind.size(0) for group in groups)
    pin = (
        pin_memory
        and next(iter(devices)).type == "cpu"
        and torch.cuda.is_available()
    )
    combined: dict[str, Tensor | int] = {}
    for field in fields(groups[0]):
        if field.name == "prompt_length":
            combined[field.name] = min(group.prompt_length for group in groups)
            continue
        if field.name == "statistics_refreshed":
            combined[field.name] = all(
                group.statistics_refreshed for group in groups
            )
            continue
        values = [getattr(group, field.name) for group in groups]
        if not all(isinstance(value, Tensor) for value in values):
            raise TypeError(f"unexpected non-tensor rollout field {field.name}")
        first = values[0]
        if first.dim() >= 2:
            if not all(
                value.dim() == first.dim()
                and value.shape[2:] == first.shape[2:]
                and value.size(1) == group.stream_length
                for value, group in zip(values, groups, strict=True)
            ):
                raise ValueError(
                    f"rollout field {field.name} has incompatible stream shapes"
                )
            fill = PAD_SLOT if field.name == "kind" else 0
            output = torch.full(
                (total_rows, max_stream, *first.shape[2:]),
                fill,
                dtype=first.dtype,
                device=first.device,
                pin_memory=pin,
            )
            row_start = 0
            for value in values:
                row_end = row_start + value.size(0)
                output[row_start:row_end, : value.size(1)].copy_(value)
                row_start = row_end
            combined[field.name] = output
        elif first.dim() == 1:
            if not all(value.dim() == 1 for value in values):
                raise ValueError(
                    f"rollout field {field.name} has incompatible row shapes"
                )
            rows = torch.cat(values)
            combined[field.name] = rows.pin_memory() if pin else rows
        else:
            raise ValueError(f"unsupported scalar rollout field {field.name}")
    return LatentRolloutBatch(**combined)


def scatter_replay_statistics(
    packed: LatentRolloutBatch,
    groups: Sequence[LatentRolloutBatch],
) -> None:
    """Copy refreshed packed behavior/value statistics into compact groups.

    This performs four minibatch-level device transfers rather than moving a
    complete packed batch (including its duplicate sampled thoughts) back to
    host memory. Compact groups remain the canonical frozen behavior pool and
    can later be repacked in the identical order for exact PPO replay.
    """
    if not groups:
        raise ValueError("at least one rollout group is required")
    if packed.kind.size(0) != sum(group.kind.size(0) for group in groups):
        raise ValueError("packed rows do not match rollout groups")
    target_device = groups[0].kind.device
    if any(group.kind.device != target_device for group in groups):
        raise ValueError("rollout groups must share one device")
    statistic_names = (
        "old_gate_logprobs",
        "old_token_logprobs",
        "old_thought_logprobs",
        "old_thought_means",
        "old_thought_log_sigmas",
        "old_values",
    )
    statistics = {
        name: getattr(packed, name).to(target_device)
        for name in statistic_names
    }
    row_start = 0
    for group in groups:
        row_end = row_start + group.kind.size(0)
        for name, packed_value in statistics.items():
            setattr(
                group,
                name,
                packed_value[
                    row_start:row_end, : group.stream_length
                ].clone(),
            )
        # The groups now carry the packed batch's statistics, so they share
        # its refresh state.
        group.statistics_refreshed = packed.statistics_refreshed
        row_start = row_end


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
    pad_scale = (batch.kind != PAD_SLOT)[..., None].to(token_latent.dtype)
    if batch.thoughts.size(-1) == 0:
        # Pinned-EMIT rollouts store zero-width thoughts and contain no
        # THOUGHT slots; the adapter cannot consume a zero-width input.
        return token_latent * pad_scale
    think_mask = batch.kind == THOUGHT_SLOT
    # The kind-select makes dense adapter evaluation value/gradient-equivalent
    # to boolean-index assignment. Unlike the latter, it has static output
    # shapes and keeps the full replay trunk inside one Inductor graph.
    thought_latent = wrapper.adapter(batch.thoughts).to(token_latent.dtype)
    inputs = torch.where(think_mask[..., None], thought_latent, token_latent)
    return inputs * pad_scale


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
    exact — behavior-age-0 PPO ratios are one by construction.
    """
    stream_inputs, beliefs = replay_beliefs(wrapper, batch)
    predicted = wrapper.thought_mean(beliefs)
    token_targets = torch.zeros_like(batch.token_ids)
    token_targets[:, :-1] = batch.token_ids[:, 1:]
    return beliefs, predicted, stream_inputs, token_targets


def compact_emit_token_logprobs(
    backbone, emit_features: Tensor, emit_targets: Tensor
) -> Tensor:
    """log P(target token) at each compact EMIT slot.

    Refresh and the trainer's update step both come through this one helper
    so their eager forwards stay bit-identical (the behavior-age-0 zero-clip
    canary). Its vocabulary-wide temporaries scale with slots x vocab; the
    replay planner's slot budget bounds that, not this function.
    """
    logits = backbone.logits_from_features(emit_features)
    return (
        logits.float()
        .log_softmax(-1)
        .gather(-1, emit_targets[..., None])
        .squeeze(-1)
    )


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
    slot_budget: int | None = None,
):
    """Yield stable length-sorted replay shards under B*L^2 and B*L budgets.

    Causal full-stream replay is governed by two independent memory terms:
    attention area (B*L^2, quadratic) and the vocabulary head over the
    shard's slots (B*L x 50257-wide logits plus their autograd-retained
    log-softmax — the LINEAR term, and the larger one for fat shards of
    ordinary-length trajectories). ``slot_budget`` bounds the linear term;
    without it, raising ``attention_budget`` alone lets short-L shards grow
    their slot count unboundedly and the emit-logits pass OOMs before
    attention does. Stable sorting groups similarly sized independent
    trajectories; rare 1K-4K outliers automatically receive smaller shards.
    ``rows`` maps each compact shard back into the parent batch for
    refresh-stat writes. The final element repeats those row indices as the
    host-side Python list they were built from, so callers can make
    per-shard branch decisions (has-THINK, has-EMIT) against a once-per-batch
    CPU table instead of a blocking device sync inside every shard.
    """
    if max_trajectories < 1:
        raise ValueError("replay max trajectories must be positive")
    if attention_budget < 1:
        raise ValueError("replay attention budget must be positive")
    if bucket_multiple < 1:
        raise ValueError("replay bucket multiple must be positive")
    if slot_budget is not None and slot_budget < 1:
        raise ValueError("replay slot budget must be positive")
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
            or (
                slot_budget is not None
                and candidate_rows * candidate_length > slot_budget
            )
        )
        if shard and exceeds:
            row_tensor = torch.tensor(
                shard, dtype=torch.long, device=batch.kind.device
            )
            yield (
                select_trajectory_rows(batch, row_tensor, shard_length),
                row_tensor,
                shard_length,
                shard,
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
            shard,
        )


def refresh_old_statistics(
    wrapper: LatentThoughtModel,
    critic,
    batch: LatentRolloutBatch,
    max_trajectories: int = 32,
    attention_budget: int = 4 * 1024 * 1024,
    bucket_multiple: int = 1,
    slot_budget: int | None = None,
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
    reintroducing exactly the behavior-age-0 ratio drift it exists to remove.
    Eagerly the numerics are identical either way; the cost is one
    forward's transient activation memory.

    ``old_values`` come from the separate critic model — the rollout itself
    never computes values, so this is where GAE's baseline is filled in.
    Replay is stably length-sorted and split only across independent
    trajectories under a B*L^2 budget, bounding quadratic attention memory
    without changing any consumed statistic. The update path uses the same
    deterministic planner so compiled refresh/update forwards remain
    numerically identical for the first behavior minibatch.
    """
    backbone = wrapper.backbone
    # Pinned-EMIT batches keep zero-width thoughts; their old_thought_logprobs
    # then stay zero-width too, and the thought refresh below is skipped.
    if batch.old_thought_logprobs.shape[-1] == 0:
        batch.old_thought_logprobs = torch.zeros_like(batch.thoughts)
    if batch.old_thought_means.shape[-1] == 0:
        batch.old_thought_means = torch.zeros_like(batch.thoughts)
    if batch.old_thought_log_sigmas.shape[-1] == 0:
        batch.old_thought_log_sigmas = torch.zeros_like(batch.thoughts)
    for microbatch, rows, stream_length, _ in iter_length_aware_microbatches(
        batch, max_trajectories, attention_budget, bucket_multiple,
        slot_budget=slot_budget,
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
        # The grad-mode compile-guard argument above covers only the compiled
        # replay_head_inputs; this tail is eager, where grad mode changes no
        # forward kernel. Dropping its (discarded) graph keeps the retained
        # log-softmax outputs — ~0.3 KB per EMIT slot times the vocabulary —
        # out of the refresh peak.
        with torch.no_grad():
            emit_features = wrapper.renderer_features(
                stream_inputs[emit_mask], beliefs[emit_mask]
            )
            compact_token_logprobs = compact_emit_token_logprobs(
                backbone, emit_features, token_targets[emit_mask]
            )
        token_logprobs = torch.zeros_like(microbatch.old_token_logprobs)
        token_logprobs[emit_mask] = compact_token_logprobs
        if microbatch.thoughts.size(-1):
            # The thought decided at gate position p is stored at p+1 — the
            # same shift as token targets — so per-dim log-probs align with
            # think_mask.
            thought_means, thought_targets, think_mask = select_thought_actions(
                microbatch, predicted
            )
            thought_log_sigma = wrapper.transition.predict_log_sigma(
                beliefs[think_mask]
            )
            compact_thought_logprobs = wrapper.transition.per_dim_log_prob(
                thought_targets,
                thought_means,
                thought_log_sigma,
            ).float()
            thought_logprobs = torch.zeros_like(microbatch.old_thought_logprobs)
            thought_logprobs[think_mask] = compact_thought_logprobs
            # Behavior Gaussian parameters come from the SAME replay forward
            # as the log-probabilities so the age-0 canary extends to the
            # projected objective: Mahalanobis distance is exactly zero on
            # fresh behavior.
            thought_mean_statistics = torch.zeros_like(
                microbatch.old_thought_means
            )
            thought_mean_statistics[think_mask] = thought_means.detach().float()
            thought_log_sigma_statistics = torch.zeros_like(
                microbatch.old_thought_log_sigmas
            )
            thought_log_sigma_statistics[think_mask] = (
                thought_log_sigma.detach().float()
            )
        else:
            thought_logprobs = torch.zeros_like(microbatch.old_thought_logprobs)
            thought_mean_statistics = torch.zeros_like(
                microbatch.old_thought_means
            )
            thought_log_sigma_statistics = torch.zeros_like(
                microbatch.old_thought_log_sigmas
            )
        with torch.no_grad():
            # Advanced row indexing materializes a copy, so assignment must
            # target the parent explicitly (``view.copy_`` would update only
            # the temporary).
            batch.old_values[rows, :stream_length] = values
            batch.old_gate_logprobs[rows, :stream_length] = gate_logprobs
            batch.old_token_logprobs[rows, :stream_length] = token_logprobs
            batch.old_thought_logprobs[rows, :stream_length] = thought_logprobs
            batch.old_thought_means[rows, :stream_length] = (
                thought_mean_statistics
            )
            batch.old_thought_log_sigmas[rows, :stream_length] = (
                thought_log_sigma_statistics
            )
    batch.statistics_refreshed = True
