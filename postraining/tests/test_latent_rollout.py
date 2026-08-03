from __future__ import annotations

import copy
import math
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import train_gpt as baseline
from pretraining.fresh_lejepa.fresh_lejepa_train import FreshLeJEPAGPT
from pretraining.fresh_lejepa.fresh_lejepa_train_v1_probe_shared_rms_pope import FreshLeJEPASharedRMSV1PoPE
from postraining.core import (
    generalized_advantage_estimate,
    nearby_numeric_reward,
)
from postraining.latent_rollout import (
    PAD_SLOT,
    TOKEN_SLOT,
    LatentRolloutBatch,
    assemble_stream_latents,
    assign_terminal_rewards,
    build_replay_plan,
    compact_emit_token_logprobs,
    compact_next_slots,
    compact_slots,
    pack_rollout_groups_for_replay,
    continuation_reward,
    emitted_token_rows,
    generated_slot_mask,
    iter_length_aware_microbatches,
    iter_planned_replay_microbatches,
    plan_length_aware_shards,
    refresh_old_statistics,
    replay_beliefs,
    replay_head_inputs,
    scatter_replay_statistics,
    scatter_slots,
    select_trajectory_rows,
    slot_index,
    rollout_continuations,
    split_rollout_groups,
    trim_stream,
)
from postraining.latent_thought import (
    THOUGHT_INPUT_SCHEMA,
    DecodeRangeMask,
    LatentThoughtModel,
)
from postraining.nano_backbone import NanoGPTBackbone
from postraining.model_io import _pope_construction
from postraining.vapo.config import (
    POLAR_EXPRESS_STEP_COMPENSATION,
    build_arg_parser,
    validate_args,
)
from postraining.train_latent_vapo import (
    MathPromptSampler,
    REWARD_SCHEMA,
    build_optimizers,
    evaluate_aime_latent,
    lockstep_decode_metrics,
    math_dataset_identity,
    measure_post_update_policy_drift,
    sample_prompt_batch,
    score_math_rollout,
    save_checkpoint,
    update_minibatch,
)
from postraining.latent_eval import verify_terminated_answer
from postraining.vapo.schemas import (
    ACTOR_OBJECTIVE_SCHEMA,
    EXECUTION_SCHEMA,
    REPLAY_NUMERICS_SCHEMA,
    execution_schema_for_rollout_scheduler,
    resume_execution_schema_compatible,
    resume_replay_schema_compatible,
    value_support_geometry_matches,
)
from postraining.value_model import SeparateCritic

KWARGS = dict(
    vocab_size=32, num_layers=3, model_dim=32, num_heads=4, num_kv_heads=2,
    mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.01,
    logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
)


def _wrapper(seed: int = 3) -> LatentThoughtModel:
    torch.manual_seed(seed)
    with _pope_construction():
        backbone = FreshLeJEPASharedRMSV1PoPE(**KWARGS).eval()
    # Full-model RL: everything trains except the unused critic probe,
    # mirroring the trainer's setup exactly.
    wrapper = LatentThoughtModel(backbone)
    for parameter in wrapper.parameters():
        parameter.requires_grad_(True)
    for parameter in backbone.critic_probe.parameters():
        parameter.requires_grad_(False)
    return wrapper


def _bf16_wrapper(seed: int = 3) -> LatentThoughtModel:
    """A tiny model in a mixed-precision regime all-fp32 models can't exercise.

    The backbone body is bf16 with CastedLinear modules and low-dim params
    restored to fp32 while the new heads stay fp32.  ``load_model`` now keeps
    fp32 masters (a whole-body bf16 cast measurably degrades the PoPE
    checkpoint), but the rollout/replay stack must stay dtype-robust: the
    fp32 adapter crashing on a bf16 operand was a real blocker this pins.
    """
    torch.manual_seed(seed)
    with _pope_construction():
        backbone = FreshLeJEPASharedRMSV1PoPE(**KWARGS).bfloat16()
        for module in backbone.modules():
            if isinstance(module, baseline.CastedLinear):
                module.float()
        baseline.restore_low_dim_params_to_fp32(backbone)
    backbone.eval()
    wrapper = LatentThoughtModel(backbone)
    for parameter in wrapper.parameters():
        parameter.requires_grad_(True)
    for parameter in backbone.critic_probe.parameters():
        parameter.requires_grad_(False)
    return wrapper


def _critic(seed: int = 11) -> SeparateCritic:
    torch.manual_seed(seed)
    with _pope_construction():
        trunk = FreshLeJEPASharedRMSV1PoPE(**KWARGS).eval()
    critic = SeparateCritic(trunk, num_bins=17, sigma_ratio=2.0).eval()
    # The v215 head init (zero weight, prior bias) makes every value the
    # constant prior; de-zero the weight so values are input-dependent and
    # the exactness assertions below carry weight.
    with torch.no_grad():
        critic.head.weight.normal_(std=0.05)
    return critic


def _rollout(wrapper, batch=2, prompt=5, new_tokens=4, stream_steps=None, seed=7):
    prompt_ids = torch.randint(0, 32, (batch, prompt))
    generator = torch.Generator().manual_seed(seed)
    result = rollout_continuations(
        wrapper, prompt_ids, new_tokens, stream_steps or 8 * new_tokens,
        1.0, 1.0, generator=generator,
    )
    return trim_stream(result)


def _optimizers(wrapper, critic, learning_rate=1e-3):
    """The trainer's actual optimizer layout, at test-scale rates (CPU: unfused)."""
    return build_optimizers(
        wrapper, critic, learning_rate=learning_rate,
        fused=False,
    )


def test_continuation_reward_bounds_and_ordering():
    assert continuation_reward("hello world", "hello world") == 1.0
    assert continuation_reward("", "hello") == 0.0
    assert continuation_reward("xyz", "abc") == 0.0
    partial = continuation_reward("hello there", "hello world")
    assert 0.0 < partial < 1.0
    # A longer matching prefix scores strictly higher at equal overlap.
    assert continuation_reward("ab__", "abcd") > continuation_reward("__ab", "abcd")


def test_rollout_emits_exactly_the_requested_tokens():
    wrapper = _wrapper()
    batch = _rollout(wrapper, batch=3, prompt=4, new_tokens=5)
    for row in emitted_token_rows(batch):
        assert len(row) == 5


def test_stream_budget_smaller_than_emit_cap_is_rejected():
    wrapper = _wrapper()
    with pytest.raises(ValueError):
        rollout_continuations(wrapper, torch.randint(0, 32, (1, 4)), 8, 4, 1.0, 1.0)


def test_stream_storage_is_internally_consistent():
    wrapper = _wrapper()
    batch = _rollout(wrapper, batch=4)
    prompt = batch.prompt_length
    assert torch.all(batch.kind[:, :prompt] == TOKEN_SLOT)
    # The first action decides at the last prompt slot; its consequence (the
    # first generated token, with the belief that produced it) lands at the
    # next slot — the codebase's "+1 shift" convention.
    assert torch.all(batch.action_mask[:, prompt - 1] == 1)
    assert torch.all(batch.kind[:, prompt] == TOKEN_SLOT)
    # Every recorded action produced a generated-token slot right after it,
    # which is exactly where the hasThought flag derives from.
    action_positions = batch.action_mask.nonzero()
    carried = generated_slot_mask(batch)
    for row, position in action_positions.tolist():
        assert int(batch.kind[row, position + 1]) == TOKEN_SLOT
        assert bool(carried[row, position + 1])
    # Prompt slots and the first action slot carry no hidden.
    assert not carried[:, :prompt].any()
    # PAD slots carry no hiddens, no actions, no rewards.
    pads = batch.kind == PAD_SLOT
    assert float(batch.action_mask[pads].sum()) == 0.0
    assert float(batch.hiddens[pads].abs().sum()) == 0.0


def test_emit_token_logprobs_helper_matches_the_explicit_readout():
    """The fused helper is the readout tail, spelled out.

    The trainer hands this function to one ``torch.compile`` artifact shared
    by refresh and update, so the whole softcap/log-softmax/gather chain now
    lives behind a single call. Pin it against the formula written out longhand
    so a fusion can never quietly change what is being computed.
    """
    wrapper = _wrapper()
    batch = _rollout(wrapper, batch=2, prompt=6, new_tokens=4)
    emit_index = slot_index(batch.action_mask.bool())
    with torch.no_grad():
        stream_inputs, beliefs = replay_beliefs(wrapper, batch)
        emit_inputs = compact_slots(stream_inputs, emit_index)
        emit_beliefs = compact_slots(beliefs, emit_index)
        targets = compact_next_slots(batch.token_ids, emit_index)
        expected = (
            wrapper.backbone.logits_from_features(
                wrapper.renderer_features(emit_inputs, emit_beliefs)
            )
            .float()
            .log_softmax(-1)
            .gather(-1, targets[..., None])
            .squeeze(-1)
        )
        actual = compact_emit_token_logprobs(
            wrapper, emit_inputs, emit_beliefs, targets
        )
    assert actual.shape == (emit_index.numel(),)
    assert torch.equal(actual, expected)


def test_compiled_emit_token_logprobs_agrees_with_eager():
    """Compiling the readout tail must not move the value it produces.

    Refresh and update share one artifact, so a fusion cannot desynchronize
    them relative to each other — but it can move both away from the eager
    reference, and the PPO ratio is a difference of two log-probabilities
    where a shifted readout shows up directly as clip fraction. Inductor is
    free to reorder the vocabulary reduction, so this is a closeness bound
    rather than bit equality.
    """
    wrapper = _wrapper()
    batch = _rollout(wrapper, batch=2, prompt=6, new_tokens=4)
    emit_index = slot_index(batch.action_mask.bool())
    with torch.no_grad():
        stream_inputs, beliefs = replay_beliefs(wrapper, batch)
        emit_inputs = compact_slots(stream_inputs, emit_index)
        emit_beliefs = compact_slots(beliefs, emit_index)
        targets = compact_next_slots(batch.token_ids, emit_index)
        eager = compact_emit_token_logprobs(
            wrapper, emit_inputs, emit_beliefs, targets
        )
        compiled = torch.compile(
            compact_emit_token_logprobs, fullgraph=True, dynamic=True
        )(wrapper, emit_inputs, emit_beliefs, targets)
    torch.testing.assert_close(compiled, eager, rtol=1e-5, atol=1e-6)


def test_replay_reproduces_rollout_logprobs():
    wrapper = _wrapper()
    batch = _rollout(wrapper, batch=2, prompt=6, new_tokens=4)
    backbone = wrapper.backbone
    with torch.no_grad():
        stream_inputs, beliefs = replay_beliefs(wrapper, batch)
        features = wrapper.renderer_features(stream_inputs, beliefs)
        logits = backbone.logits_from_features(features).float()
        token_targets = torch.zeros_like(batch.token_ids)
        token_targets[:, :-1] = batch.token_ids[:, 1:]
        token_logprobs = logits.log_softmax(-1).gather(
            -1, token_targets[..., None]
        ).squeeze(-1)
    # The rollout never values: old_values stay zero until the separate
    # critic fills them in refresh_old_statistics.
    assert float(batch.old_values.abs().sum()) == 0.0
    actions = batch.action_mask.bool()
    torch.testing.assert_close(
        token_logprobs[actions], batch.old_token_logprobs[actions],
        rtol=2e-4, atol=2e-4,
    )


def test_assembled_latents_zero_pads_and_route_hiddens_through_combiner():
    wrapper = _wrapper()
    with torch.no_grad():
        # A fresh combiner is an exact identity; give it a live content
        # channel so the carried-hidden routing is observable.
        wrapper.combiner.carry.weight.normal_(std=0.02)
        wrapper.combiner.type_bias.normal_(std=0.02)
    batch = _rollout(wrapper)
    with torch.no_grad():
        latents = assemble_stream_latents(wrapper, batch)
    pads = batch.kind == PAD_SLOT
    assert float(latents[pads].abs().sum()) == 0.0
    carried = generated_slot_mask(batch)
    token_latent = wrapper.embed_tokens(batch.token_ids)
    with torch.no_grad():
        expected = wrapper.combiner(token_latent, batch.hiddens, carried)
    torch.testing.assert_close(latents[carried], expected[carried])
    # Uncarried non-pad slots are bitwise the plain token path.
    plain = ~carried & (batch.kind != PAD_SLOT)
    assert torch.equal(latents[plain], token_latent[plain])


def test_discarded_carry_batches_refuse_replay():
    """Zero-width hiddens are replayable only for pinned token-only modes.

    A latent rollout with ``replay_storage=False`` injected carries the
    stored stream no longer contains; replaying it would silently rebuild
    plain token inputs the behavior policy never saw, so it must raise.
    """
    wrapper = _wrapper()
    prompt_ids = torch.randint(0, 32, (2, 5))
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        discarded = trim_stream(
            rollout_continuations(
                wrapper, prompt_ids, 3, 24, 1.0, 1.0,
                generator=generator, replay_storage=False,
            )
        )
    assert discarded.carry_injected and discarded.hiddens.size(-1) == 0
    with pytest.raises(ValueError, match="discarded its carried hiddens"):
        assemble_stream_latents(wrapper, discarded)
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        pinned = trim_stream(
            rollout_continuations(
                wrapper, prompt_ids, 3, 24, 1.0, 1.0,
                generator=generator, replay_storage=False, pin_emit=True,
            )
        )
        assert not pinned.carry_injected
        # Pinned token-only rollouts replay through the plain token path.
        assemble_stream_latents(wrapper, pinned)


def test_terminal_reward_lands_on_the_last_action():
    wrapper = _wrapper()
    batch = _rollout(wrapper, batch=3)
    scores = torch.tensor([0.25, 0.5, 0.75])
    assign_terminal_rewards(batch, scores)
    torch.testing.assert_close(batch.rewards.sum(1), scores)
    for row in range(3):
        position = int(batch.rewards[row].nonzero()[0])
        assert float(batch.action_mask[row, position]) == 1.0
        assert float(batch.action_mask[row, position + 1 :].sum()) == 0.0


def test_update_minibatch_trains_the_full_policy_model():
    wrapper = _wrapper()
    critic = _critic()
    backbone = wrapper.backbone
    with torch.no_grad():
        backbone.policy_probe.output.weight.normal_(std=0.02)
    batch = _rollout(wrapper, batch=4, prompt=5, new_tokens=3)
    assign_terminal_rewards(batch, torch.rand(4))
    refresh_old_statistics(wrapper, critic, batch)
    # Fresh trunks zero-init output projections, so at step one the gradient
    # reaches attn.proj (whose input is nonzero) but not yet c_qkv behind it.
    trunk_before = backbone.blocks[0].attn.proj.weight.clone()
    embed_before = backbone.tok_emb.weight.clone()
    critic_trunk_before = critic.trunk.blocks[0].attn.proj.weight.clone()
    combiner_carry_before = wrapper.combiner.carry.weight.clone()
    frozen_probe_before = backbone.critic_probe.output.weight.clone()
    metrics = update_minibatch(wrapper, critic, batch, _optimizers(wrapper, critic))
    assert all(
        torch.isfinite(torch.tensor(value)) for value in metrics.values()
    ), metrics
    # v2 full-model RL: the policy trunk and embeddings MUST move.
    assert not torch.equal(backbone.blocks[0].attn.proj.weight, trunk_before)
    assert not torch.equal(backbone.tok_emb.weight, embed_before)
    # The separate critic is fully trainable: value CE must reach its trunk.
    assert not torch.equal(
        critic.trunk.blocks[0].attn.proj.weight, critic_trunk_before
    )
    # The zero-init carry matrix has a full-rank first-step gradient
    # (loss direction outer hidden), so it must move on the first update.
    assert not torch.equal(wrapper.combiner.carry.weight, combiner_carry_before)
    # Only the unused backbone critic probe stays frozen.
    assert torch.equal(backbone.critic_probe.output.weight, frozen_probe_before)
    assert metrics["trunk_grad_norm"] > 0.0


def _decode_group(row_actions: list[int], prompt: int, bucket: int = 1):
    rows = len(row_actions)
    # trim_stream rounds the kept length up to a replay bucket and pads past
    # the original stream, so stream_length is deliberately NOT the row length.
    used = prompt + max(row_actions)
    stream = -(-used // bucket) * bucket
    zeros = torch.zeros(rows, stream)
    action_mask = zeros.clone()
    for row, count in enumerate(row_actions):
        action_mask[row, prompt - 1 : prompt - 1 + count] = 1.0
    return LatentRolloutBatch(
        kind=torch.full((rows, stream), TOKEN_SLOT, dtype=torch.long),
        token_ids=torch.zeros((rows, stream), dtype=torch.long),
        hiddens=torch.zeros(rows, stream, 0),
        action_mask=action_mask,
        old_token_logprobs=zeros.clone(),
        old_values=zeros.clone(),
        rewards=zeros.clone(),
        reward_scalar=torch.zeros(rows),
        prompt_length=prompt,
    )


def test_lockstep_decode_metrics_price_a_chunk_at_its_longest_trajectory():
    # Two chunks of two groups: the chunk pays for its longest ROW, not its
    # mean, which is exactly the lockstep waste the metric has to expose.
    def pool(bucket: int) -> list:
        return [
            _decode_group([3, 2], prompt=2, bucket=bucket),
            _decode_group([5, 4], prompt=4, bucket=bucket),
            _decode_group([2, 2], prompt=3, bucket=bucket),
            _decode_group([1, 1], prompt=2, bucket=bucket),
        ]

    metrics = lockstep_decode_metrics(pool(bucket=1), chunk_groups=2)
    assert metrics["decode_steps_per_chunk_max"] == 5.0
    assert metrics["decode_steps_per_chunk_mean"] == pytest.approx(3.5)
    # 20 actions over 8 rows is 2.5 kept steps per row against 3.5 paid.
    assert metrics["decode_step_utilization"] == pytest.approx(2.5 / 3.5)
    # The pool reaches this function through trim_stream, which rounds every
    # group's stream up to a replay bucket. Reading the length off
    # stream_length instead of action_mask made the metric swing by up to a
    # full bucket for identical decoding -- fatal for a metric meant to
    # detect a change. Bucketing must move nothing.
    for bucket in (8, 64):
        assert lockstep_decode_metrics(
            pool(bucket=bucket), chunk_groups=2
        ) == metrics
    # One group per chunk is the sequential rollout path: nothing is wasted
    # beyond each group's own ragged rows.
    solo = lockstep_decode_metrics(pool(bucket=64), chunk_groups=1)
    assert solo["decode_steps_per_chunk_mean"] == pytest.approx(11 / 4)
    assert solo["decode_steps_per_chunk_max"] == 5.0
    with pytest.raises(ValueError, match="at least one rollout group"):
        lockstep_decode_metrics([], chunk_groups=2)


def test_lockstep_decode_metrics_survive_real_replay_bucketing():
    # The synthetic case above pins the arithmetic; this pins it against an
    # actual rollout put through the actual trim_stream, which is where the
    # bucketing bug came from.
    wrapper = _wrapper()
    batch = _rollout(wrapper, batch=6, prompt=5, new_tokens=9)
    row_actions = batch.action_mask.sum(1)
    baseline = lockstep_decode_metrics([batch], chunk_groups=1)
    assert baseline["decode_steps_per_chunk_max"] == float(row_actions.max())
    for multiple in (1, 8, 64, 128):
        trimmed = trim_stream(batch, multiple=multiple)
        assert lockstep_decode_metrics([trimmed], chunk_groups=1) == baseline
    # ...and the quantity the first version read really does move, so the
    # test above is not vacuous.
    assert (
        trim_stream(batch, multiple=128).stream_length
        != trim_stream(batch, multiple=1).stream_length
    )


def test_update_minibatch_names_the_shard_of_a_non_finite_actor_loss():
    wrapper = _wrapper()
    critic = _critic()
    batch = _rollout(wrapper, batch=4, prompt=5, new_tokens=3)
    assign_terminal_rewards(batch, torch.rand(4))
    refresh_old_statistics(wrapper, critic, batch)
    # A poisoned behavior likelihood makes the policy ratio non-finite. The
    # guard is resolved once per minibatch rather than before each backward,
    # so it must still abort -- and still name the shard and the term.
    batch.old_token_logprobs.fill_(float("nan"))
    trunk_before = wrapper.backbone.blocks[0].attn.proj.weight.clone()
    critic_before = critic.trunk.blocks[0].attn.proj.weight.clone()
    with pytest.raises(RuntimeError, match="non-finite loss") as failure:
        update_minibatch(wrapper, critic, batch, _optimizers(wrapper, critic))
    assert "shard=0 actor" in str(failure.value)
    assert "policy=nan" in str(failure.value)
    # The guard now fires after the backward passes, so what it has to protect
    # is the weights, not the gradients: nothing may have stepped.
    assert torch.equal(
        wrapper.backbone.blocks[0].attn.proj.weight, trunk_before
    )
    assert torch.equal(critic.trunk.blocks[0].attn.proj.weight, critic_before)


def test_update_minibatch_names_a_non_finite_value_loss_without_an_actor():
    wrapper = _wrapper()
    critic = _critic()
    batch = _rollout(wrapper, batch=4, prompt=5, new_tokens=3)
    assign_terminal_rewards(batch, torch.rand(4))
    refresh_old_statistics(wrapper, critic, batch)
    with torch.no_grad():
        critic.head.weight.fill_(float("nan"))
    # value_only skips the actor half of every shard and returns through its
    # own step_optimizers call, so the deferred guard must still catch a critic
    # that has gone non-finite on its own -- and still catch it first.
    critic_before = critic.trunk.blocks[0].attn.proj.weight.clone()
    with pytest.raises(RuntimeError, match="non-finite loss") as failure:
        update_minibatch(
            wrapper,
            critic,
            batch,
            _optimizers(wrapper, critic),
            value_only=True,
        )
    assert "shard=0 value" in str(failure.value)
    assert torch.equal(critic.trunk.blocks[0].attn.proj.weight, critic_before)


def test_refresh_old_statistics_matches_the_update_code_path_exactly():
    wrapper = _wrapper()
    critic = _critic()
    with torch.no_grad():
        # A live combiner makes the carried-hidden path load-bearing; a fresh
        # identity combiner would let this pass even if refresh dropped it.
        wrapper.combiner.carry.weight.normal_(std=0.02)
    batch = _rollout(wrapper, batch=2, prompt=6, new_tokens=4)
    refresh_old_statistics(wrapper, critic, batch)
    backbone = wrapper.backbone
    with torch.no_grad():
        beliefs, stream_inputs = replay_head_inputs(
            wrapper, batch
        )
        values = critic.values(batch).float()
        action_mask = batch.action_mask.bool()
        action_features = wrapper.renderer_features(
            stream_inputs[action_mask], beliefs[action_mask]
        )
        compact_token_logprobs = (
            backbone.logits_from_features(action_features)
            .float()
            .log_softmax(-1)
            .gather(
                -1,
                compact_next_slots(
                    batch.token_ids, slot_index(action_mask)
                )[..., None],
            )
            .squeeze(-1)
        )
        token_logprobs = torch.zeros_like(batch.old_token_logprobs)
        token_logprobs[action_mask] = compact_token_logprobs
    assert torch.equal(batch.old_values, values)
    assert torch.equal(batch.old_token_logprobs, token_logprobs)


def test_microbatched_refresh_matches_full_group_refresh():
    wrapper = _wrapper()
    critic = _critic()
    batch = _rollout(wrapper, batch=5, prompt=6, new_tokens=4)
    full = copy.deepcopy(batch)
    microbatched = copy.deepcopy(batch)

    refresh_old_statistics(
        wrapper, critic, full, max_trajectories=5,
        attention_budget=10**9,
    )
    refresh_old_statistics(
        wrapper, critic, microbatched, max_trajectories=2,
        attention_budget=10**9,
    )

    masks = {
        "old_values": batch.action_mask.bool(),
        "old_token_logprobs": batch.action_mask.bool(),
    }
    for name, mask in masks.items():
        torch.testing.assert_close(
            getattr(microbatched, name)[mask],
            getattr(full, name)[mask],
            rtol=1e-6,
            atol=1e-7,
        )


def test_length_aware_replay_planner_sorts_and_respects_attention_area():
    wrapper = _wrapper()
    batch = rollout_continuations(
        wrapper,
        torch.randint(0, 32, (5, 4)),
        max_new_tokens=4,
        max_stream_steps=20,
        temperature=1.0,
        top_p=1.0,
    )
    # Give rows deliberately non-monotonic used lengths without changing the
    # parent row order. Replay sorting must be stable and local to the plan.
    lengths = [9, 17, 12, 20, 10]
    for row, length in enumerate(lengths):
        batch.kind[row, :length] = TOKEN_SLOT
        batch.kind[row, length:] = PAD_SLOT
    budget = 2 * 16 * 16
    shards = list(
        iter_length_aware_microbatches(
            batch,
            max_trajectories=5,
            attention_budget=budget,
            bucket_multiple=4,
        )
    )
    planned_rows = [
        row for _, rows, _, _ in shards for row in rows.tolist()
    ]
    assert planned_rows == [3, 1, 2, 4, 0]
    for microbatch, rows, stream_length, host_rows in shards:
        assert microbatch.kind.shape == (rows.numel(), stream_length)
        assert rows.numel() == 1 or rows.numel() * stream_length**2 <= budget
        # The host-side row list mirrors the device index tensor so
        # per-shard branch guards never need a device sync.
        assert host_rows == rows.tolist()


def test_replay_shards_weakly_mark_the_stream_dimension_dynamic():
    """Every shard leaves the planner with a weakly dynamic stream dim.

    Without the mark, a first traced shard whose bucketed length equals the
    model width duck-types the stream symbol onto the thought width and the
    NEXT length pays a full recompile. The mark must land on the shard
    tensors the planner yields, since that is the single path refresh and
    update share.
    """
    wrapper = _wrapper()
    batch = rollout_continuations(
        wrapper,
        torch.randint(0, 32, (4, 4)),
        max_new_tokens=4,
        max_stream_steps=20,
        temperature=1.0,
        top_p=1.0,
    )
    for row in range(4):
        batch.kind[row, :8] = TOKEN_SLOT
        batch.kind[row, 8:] = PAD_SLOT
    shards = list(
        iter_length_aware_microbatches(
            batch, max_trajectories=4, attention_budget=10**9
        )
    )
    assert shards
    for microbatch, _, stream_length, _ in shards:
        stream_tensors = [
            value
            for field in fields(microbatch)
            if isinstance((value := getattr(microbatch, field.name)), torch.Tensor)
            and value.dim() >= 2
            and value.size(1) == stream_length
        ]
        assert stream_tensors
        for value in stream_tensors:
            # maybe_mark_dynamic records a WEAK hint: it steers the first
            # trace away from duck sizing without erroring if some later
            # guard genuinely has to specialize the dimension.
            assert 1 in getattr(value, "_dynamo_weak_dynamic_indices", set())


def test_length_aware_microbatches_slot_budget_bounds_linear_term():
    wrapper = _wrapper()
    batch = rollout_continuations(
        wrapper,
        torch.randint(0, 32, (8, 4)),
        max_new_tokens=4,
        max_stream_steps=20,
        temperature=1.0,
        top_p=1.0,
    )
    # Uniform short lengths: the quadratic budget alone would pack all 8
    # rows into one shard; the linear slot budget must split them.
    for row in range(8):
        batch.kind[row, :8] = TOKEN_SLOT
        batch.kind[row, 8:] = PAD_SLOT
    shards = list(
        iter_length_aware_microbatches(
            batch,
            max_trajectories=8,
            attention_budget=10**9,
            slot_budget=3 * 8,
        )
    )
    assert len(shards) == 3  # 3 + 3 + 2 rows
    covered = sorted(
        row for _, rows, _, _ in shards for row in rows.tolist()
    )
    assert covered == list(range(8))
    for microbatch, rows, stream_length, host_rows in shards:
        assert rows.numel() == 1 or rows.numel() * stream_length <= 3 * 8
        assert host_rows == rows.tolist()
    # A single over-budget row still ships as its own shard.
    lone = list(
        iter_length_aware_microbatches(
            batch,
            max_trajectories=8,
            attention_budget=10**9,
            slot_budget=1,
        )
    )
    assert all(rows.numel() == 1 for _, rows, _, _ in lone)


def test_length_aware_refresh_writes_noncontiguous_parent_rows():
    wrapper = _wrapper()
    critic = _critic()
    batch = _rollout(wrapper, batch=6, prompt=6, new_tokens=4)
    batch.old_values.fill_(float("nan"))
    batch.old_token_logprobs.fill_(float("nan"))
    refresh_old_statistics(
        wrapper,
        critic,
        batch,
        max_trajectories=2,
        attention_budget=10**9,
    )
    assert torch.isfinite(batch.old_values[batch.action_mask.bool()]).all()
    assert torch.isfinite(
        batch.old_token_logprobs[batch.action_mask.bool()]
    ).all()


def test_trajectory_microbatch_update_matches_full_group_objective_and_step():
    base_wrapper = _wrapper()
    base_critic = _critic()
    batch = _rollout(
        base_wrapper, batch=6, prompt=5, new_tokens=4, stream_steps=20, seed=19
    )
    scores = torch.tensor([0.9, 0.1, 0.7, 0.2, 0.8, 0.3])
    assign_terminal_rewards(batch, scores)
    refresh_old_statistics(
        base_wrapper, base_critic, batch,
        max_trajectories=batch.kind.size(0), attention_budget=10**9,
    )
    # The stored behavior statistics stay frozen while the current policy
    # moves. This makes the shard-invariance check exercise nonzero KL
    # accumulation rather than the trivial refresh-equals-current case.
    with torch.no_grad():
        renderer_output = base_wrapper.backbone.policy_probe.output.weight
        renderer_output.add_(
            torch.linspace(
                -0.01, 0.01, renderer_output.numel()
            ).reshape_as(renderer_output)
        )

    full_wrapper = copy.deepcopy(base_wrapper)
    micro_wrapper = copy.deepcopy(base_wrapper)
    full_critic = copy.deepcopy(base_critic)
    micro_critic = copy.deepcopy(base_critic)
    full_optimizers = _optimizers(full_wrapper, full_critic)
    micro_optimizers = _optimizers(micro_wrapper, micro_critic)

    full_metrics = update_minibatch(
        full_wrapper,
        full_critic,
        copy.deepcopy(batch),
        full_optimizers,
        replay_max_trajectories=6,
        replay_attention_budget=10**9,
    )
    micro_metrics = update_minibatch(
        micro_wrapper,
        micro_critic,
        copy.deepcopy(batch),
        micro_optimizers,
        replay_max_trajectories=2,
        replay_attention_budget=10**9,
    )

    for key in (
        "value_loss",
        "policy_loss",
        "advantage_mean",
        "advantage_std",
        "token_behavior_kl",
        "policy_behavior_kl_per_action",
    ):
        torch.testing.assert_close(
            torch.tensor(micro_metrics[key]),
            torch.tensor(full_metrics[key]),
            rtol=2e-5,
            atol=2e-6,
        )
    for micro_parameter, full_parameter in zip(
        micro_wrapper.parameters(), full_wrapper.parameters(), strict=True
    ):
        torch.testing.assert_close(
            # Adam amplifies near-zero gradient reduction noise into a
            # learning-rate-sized sign difference; the objectives above are
            # the exactness assertion, while parameters remain numerically
            # equivalent at optimizer precision.
            micro_parameter, full_parameter, rtol=3e-4, atol=2e-5
        )
    for micro_parameter, full_parameter in zip(
        micro_critic.parameters(), full_critic.parameters(), strict=True
    ):
        torch.testing.assert_close(
            micro_parameter, full_parameter, rtol=3e-4, atol=2e-5
        )

    assert full_metrics["token_behavior_kl"] > 0
    # With tokens as the only actions, the per-action policy KL IS the token
    # behavior KL (both divide the same sum by the same action count).
    torch.testing.assert_close(
        torch.tensor(full_metrics["policy_behavior_kl_per_action"]),
        torch.tensor(full_metrics["token_behavior_kl"]),
    )

    # Adam must see one full-group update, not one update per replay shard.
    for optimizer in micro_optimizers.values():
        steps = {
            int(state["step"])
            for state in optimizer.state.values()
            if "step" in state
        }
        assert steps == {1}


def test_compact_next_slots_rejects_a_final_column_action():
    values = torch.arange(6).reshape(2, 3)
    with pytest.raises(IndexError):
        compact_next_slots(values, torch.tensor([2]))


def test_shipped_math_evaluations_share_the_250_step_cadence():
    cli = build_arg_parser().parse_args(["--checkpoint", "c", "--output", "o"])

    assert cli.aime_every == 250
    assert cli.bench_every == 250
    # BPB is a cheaper guard with its own cadence, not a math benchmark.
    assert cli.bpb_every == 150
    assert cli.rollout_scheduler == "lockstep"


def test_rollout_only_repeats_are_benchmark_scoped(capsys):
    parser = build_arg_parser()
    valid = parser.parse_args(
        [
            "--checkpoint", "c", "--output", "o",
            "--rollout-only",
            "--rollout-only-repeats", "8",
        ]
    )
    validate_args(parser, valid)

    for argv, message in (
        (
            ["--rollout-only", "--rollout-only-repeats", "0"],
            "--rollout-only-repeats must be positive",
        ),
        (
            ["--rollout-only-repeats", "8"],
            "--rollout-only-repeats requires --rollout-only",
        ),
    ):
        args = parser.parse_args(
            ["--checkpoint", "c", "--output", "o", *argv]
        )
        with pytest.raises(SystemExit):
            validate_args(parser, args)
        assert message in capsys.readouterr().err


def test_fence_flag_validation(capsys):
    parser = build_arg_parser()
    valid = parser.parse_args(
        ["--checkpoint", "c", "--output", "o",
         "--think-tokens", "--answer-fence", "--think-min-tokens", "8"]
    )
    validate_args(parser, valid)

    for argv, message in (
        (["--answer-fence"], "--answer-fence requires --think-tokens"),
        (
            ["--think-min-tokens", "4"],
            "--think-min-tokens above 1 requires --think-tokens",
        ),
        (["--think-min-tokens", "0"], "--think-min-tokens must be at least 1"),
        (
            ["--think-tokens", "--reasoning-mode", "none"],
            "--think-tokens requires a reasoning mode",
        ),
    ):
        args = parser.parse_args(["--checkpoint", "c", "--output", "o", *argv])
        with pytest.raises(SystemExit):
            validate_args(parser, args)
        assert message in capsys.readouterr().err


def test_zero_reward_actor_freeze_flag_defaults_on(capsys):
    # Round-4 guard: optimizer minibatches whose every trajectory scored
    # zero carry no policy signal, so the actor optimizer skips them by
    # default, and a sustained all-zero-pool streak stops the run at the
    # pool boundary. The freeze condition (reward mean == 0 iff all
    # rewards zero) relies on rewards being non-negative — pinned by
    # test_nearby_numeric_reward_is_nonnegative below.
    parser = build_arg_parser()
    default = parser.parse_args(["--checkpoint", "c", "--output", "o"])
    assert default.zero_reward_actor_freeze is True
    assert default.zero_reward_stop_pools == 8
    disabled = parser.parse_args(
        ["--checkpoint", "c", "--output", "o",
         "--no-zero-reward-actor-freeze",
         "--zero-reward-stop-pools", "0"]
    )
    assert disabled.zero_reward_actor_freeze is False
    assert disabled.zero_reward_stop_pools == 0

    negative = parser.parse_args(
        ["--checkpoint", "c", "--output", "o",
         "--zero-reward-stop-pools", "-1"]
    )
    with pytest.raises(SystemExit):
        validate_args(parser, negative)
    assert (
        "--zero-reward-stop-pools must be nonnegative"
        in capsys.readouterr().err
    )


def test_nearby_numeric_reward_is_nonnegative():
    from postraining.core import nearby_numeric_reward

    for prediction in ("-1e300", "-5", "0", "5.0", "1e300", "nan", "x"):
        assert nearby_numeric_reward(prediction, "7", 0.1) >= 0.0


def test_answer_fence_prompt_rewrite():
    from postraining.core import ANSWER_CLOSE, ANSWER_OPEN, THINK_CLOSE, THINK_OPEN
    from postraining.math_prompt import (
        ANSWER_FIELD_INSTRUCTIONS,
        ANSWER_FENCE_SUFFIX,
        CHINESE_ANSWER_FIELD_INSTRUCTION,
    )
    from postraining.train_latent_vapo import rewrite_prompts_for_answer_fence

    chinese_field = CHINESE_ANSWER_FIELD_INSTRUCTION
    rows = [
        {
            "prompt": [
                {
                    "content": (
                        f"Solve it. {ANSWER_FIELD_INSTRUCTIONS[0]}\n\nWhat is "
                        f"1+1?{ANSWER_FIELD_INSTRUCTIONS[1]}"
                    )
                }
            ],
            "reward_model": {"ground_truth": "2"},
        },
        # The dapo-math-17k Chinese subset carries a third instruction
        # alongside the English pair (red-team round 2, finding A).
        {
            "prompt": [
                {
                    "content": (
                        f"Solve it. {ANSWER_FIELD_INSTRUCTIONS[0]}\n\n"
                        f"某数学问题。\n让我们一步一步地思考。{chinese_field}"
                        f"{ANSWER_FIELD_INSTRUCTIONS[1]}"
                    )
                }
            ],
            "reward_model": {"ground_truth": "3"},
        },
    ]
    rewritten = rewrite_prompts_for_answer_fence(rows)
    content = rewritten[0]["prompt"][0]["content"]
    # Every source-specific copy is removed and exactly one shared contract
    # is appended after the bare problem.
    assert content.endswith(ANSWER_FENCE_SUFFIX)
    for fence in (THINK_OPEN, THINK_CLOSE, ANSWER_OPEN, ANSWER_CLOSE):
        assert content.count(fence) == 1
    assert ANSWER_FIELD_INSTRUCTIONS[0] not in content
    assert ANSWER_FIELD_INSTRUCTIONS[1] not in content
    chinese_content = rewritten[1]["prompt"][0]["content"]
    assert chinese_field not in chinese_content
    assert "Answer:" not in chinese_content
    assert chinese_content.endswith(ANSWER_FENCE_SUFFIX)
    assert "让我们一步一步地思考。" not in chinese_content
    # Canonicalization is idempotent, including already-fenced rows.
    assert rewrite_prompts_for_answer_fence(rewritten) == rewritten
    # Originals are never mutated.
    assert ANSWER_FIELD_INSTRUCTIONS[0] in rows[0]["prompt"][0]["content"]
    # Fail-closed on prompts with no recognizable instruction: silence
    # here would train with contradictory framing.
    with pytest.raises(ValueError, match="no recognized instruction"):
        rewrite_prompts_for_answer_fence(
            [{"prompt": [{"content": "free-form question"}]}]
        )
    # Fail-closed on a SURVIVING Answer: demand next to a known one
    # (finding A: replacement counting alone let such rows through).
    with pytest.raises(ValueError, match="left an Answer: demand"):
        rewrite_prompts_for_answer_fence(
            [
                {
                    "prompt": [
                        {
                            "content": (
                                f"Solve it. {ANSWER_FIELD_INSTRUCTIONS[0]} "
                                "Output in the format Answer: \\boxed{x}."
                            )
                        }
                    ]
                }
            ]
        )


def test_answer_fence_prompt_rewrite_covers_real_datasets():
    """Every RL/eval parquet must rewrite with no surviving Answer: demand.

    The synthetic fixture above is built from the rewrite table itself, so
    it can only ever pass (red-team round 2: 20 real dapo rows carried an
    unlisted Chinese instruction the fixture could not see).
    """
    from postraining.core import load_unique_math_rows
    from postraining.core import ANSWER_CLOSE, ANSWER_OPEN, THINK_CLOSE, THINK_OPEN
    from postraining.math_prompt import (
        ANSWER_FIELD_DEMAND,
        ANSWER_FENCE_SUFFIX,
    )
    from postraining.train_latent_vapo import rewrite_prompts_for_answer_fence

    data_dir = Path(__file__).resolve().parents[1] / "data"
    datasets = [
        path
        for path in (
            data_dir / "dapo-math-17k.parquet",
            data_dir / "gsm8k_rl_prompts.parquet",
            data_dir / "aime-2024.parquet",
            data_dir / "aime-2026.parquet",
            data_dir / "aime-2026-i.parquet",
            data_dir / "aime-2026-ii.parquet",
            data_dir / "deepmind-interpolate-rl.parquet",
            data_dir / "deepmind-interpolate-rl-full.parquet",
            # The default --bench-data: a template landing here would hit
            # every run's bench eval, not just --math-data training.
            data_dir / "deepmind-interpolate-easy.parquet",
        )
        if path.exists()
    ]
    if not datasets:
        pytest.skip("no RL prompt parquets present")
    for path in datasets:
        rows = load_unique_math_rows(str(path))
        rewritten = rewrite_prompts_for_answer_fence(rows)
        # Same case-insensitive pattern the verifier parses with — a
        # surviving "answer:" would grade as a field even though a
        # case-sensitive scan would miss it.
        assert not any(
            ANSWER_FIELD_DEMAND.search(message["content"])
            for row in rewritten
            for message in row["prompt"]
        ), f"surviving Answer: demand in {path}"
        for row in rewritten:
            content = "".join(message["content"] for message in row["prompt"])
            assert content.endswith(ANSWER_FENCE_SUFFIX), path
            for fence in (THINK_OPEN, THINK_CLOSE, ANSWER_OPEN, ANSWER_CLOSE):
                assert content.count(fence) == 1, (path, fence)


def test_answer_fence_prompt_schema_rejects_legacy_offline_evaluation():
    from postraining.math_prompt import (
        ANSWER_FENCE_PROMPT_SCHEMA,
        require_answer_fence_prompt_schema,
    )

    require_answer_fence_prompt_schema(
        {}, answer_fence=False, source="plain checkpoint"
    )
    require_answer_fence_prompt_schema(
        {"answer_fence_prompt_schema": ANSWER_FENCE_PROMPT_SCHEMA},
        answer_fence=True,
        source="canonical checkpoint",
    )
    for metadata in ({}, {"answer_fence_prompt_schema": "legacy/v0"}):
        with pytest.raises(ValueError, match="silently change"):
            require_answer_fence_prompt_schema(
                metadata, answer_fence=True, source="legacy checkpoint"
            )


def test_check_think_floor_against_corpus(capsys):
    from postraining.train_latent_vapo import check_think_floor_against_corpus

    provenance = {
        "think_span_token_percentiles": {"min": 8, "p1": 12, "p50": 90}
    }
    # Floor at or below the corpus 1st percentile: silent.
    check_think_floor_against_corpus(provenance, 12)
    assert "WARNING" not in capsys.readouterr().out
    # Above p1 but under the median: the shortest taught traces fall
    # below the floor — warn.
    check_think_floor_against_corpus(provenance, 64)
    assert "1st-percentile" in capsys.readouterr().out
    # Above the median: most of the format prior is below the floor, so
    # most structurally-gated rewards would be zero — refuse.
    with pytest.raises(RuntimeError, match="median think-span"):
        check_think_floor_against_corpus(provenance, 128)
    # Provenance without the measurement (pre-round-5 SFT checkpoint):
    # warn rather than block.
    check_think_floor_against_corpus({}, 64)
    assert "cannot be checked" in capsys.readouterr().out
    # The floor's no-op default never prints.
    check_think_floor_against_corpus({}, 1)
    assert capsys.readouterr().out == ""


def test_critic_learning_rate_defaults_to_constant_actor_rate():
    parser = build_arg_parser()
    args = parser.parse_args(["--checkpoint", "c", "--output", "o"])
    validate_args(parser, args)
    assert args.learning_rate == 2e-5
    assert args.critic_learning_rate == 2e-5
    expected_muon_lr = pytest.approx(
        2e-5
        * (0.025 / 0.015)
        * POLAR_EXPRESS_STEP_COMPENSATION
    )
    assert args.muon_learning_rate == expected_muon_lr
    assert args.critic_muon_learning_rate == expected_muon_lr

    args = parser.parse_args(
        [
            "--checkpoint", "c", "--output", "o",
            "--learning-rate", "2e-5",
        ]
    )
    validate_args(parser, args)
    assert args.learning_rate == 2e-5
    assert args.critic_learning_rate == 2e-5

    args = parser.parse_args(
        [
            "--checkpoint", "c", "--output", "o",
            "--learning-rate", "2e-5",
            "--critic-learning-rate", "5e-5",
            "--muon-learning-rate", "6e-5",
            "--critic-muon-learning-rate", "7e-5",
        ]
    )
    validate_args(parser, args)
    assert args.learning_rate == 2e-5
    assert args.critic_learning_rate == 5e-5
    assert args.muon_learning_rate == 6e-5
    assert args.critic_muon_learning_rate == 7e-5


def test_continuous_refill_requires_multiple_compiled_chunks(capsys):
    parser = build_arg_parser()
    valid = parser.parse_args(
        [
            "--checkpoint", "c", "--output", "o",
            "--rollout-scheduler", "continuous_refill",
            "--rollout-groups", "32",
        ]
    )
    validate_args(parser, valid)

    for argv, message in (
        (
            ["--rollout-groups", "1"],
            "--rollout-scheduler continuous_refill requires "
            "--rollout-groups > 1",
        ),
        (
            ["--rollout-groups", "64"],
            "--rollout-scheduler continuous_refill requires more than one "
            "rollout chunk",
        ),
        (
            ["--no-rollout-compile"],
            "--rollout-scheduler continuous_refill requires --rollout-compile",
        ),
        (
            ["--rollout-tail-graph"],
            "--rollout-tail-graph is incompatible",
        ),
    ):
        args = parser.parse_args(
            [
                "--checkpoint", "c", "--output", "o",
                "--rollout-scheduler", "continuous_refill",
                *argv,
            ]
        )
        with pytest.raises(SystemExit):
            validate_args(parser, args)
        assert message in capsys.readouterr().err


def test_graph_decode_requires_the_flex_path_and_replaces_the_tail_graph(capsys):
    parser = build_arg_parser()
    base = ["--checkpoint", "c", "--output", "o", "--rollout-graph-decode"]
    validate_args(parser, parser.parse_args([*base, "--rollout-flex-decode"]))

    for argv, message in (
        # Capture needs one static row count; on the lockstep scheduler the
        # empty KV range that makes holding one affordable exists only on
        # the flex path (continuous_refill needs no extra flag — its paged
        # step is already bucket-static).
        (
            [],
            "--rollout-graph-decode with --rollout-scheduler lockstep "
            "requires --rollout-flex-decode",
        ),
        (
            ["--rollout-flex-decode", "--rollout-tail-graph"],
            "--rollout-tail-graph has nothing left to snap to",
        ),
    ):
        args = parser.parse_args([*base, *argv])
        with pytest.raises(SystemExit):
            validate_args(parser, args)
        assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--temperature", "0.8"], "--temperature must be 1"),
        (["--top-p", "0.95"], "--top-p must be 1"),
    ],
)
def test_cli_rejects_sampling_distributions_not_scored_by_ppo(
    argv, expected, capsys
):
    parser = build_arg_parser()
    args = parser.parse_args(["--checkpoint", "c", "--output", "o", *argv])
    with pytest.raises(SystemExit):
        validate_args(parser, args)
    assert expected in capsys.readouterr().err


def test_optimizer_layout_partitions_trainable_parameters_exactly_once():
    # The probes are registered under blocks[-1], so a name-prefix exclusion
    # silently duplicates them across actor groups (a real regression this
    # pins); the frozen critic probe must appear in no group at all.
    wrapper = _wrapper()
    critic = _critic()
    optimizers = _optimizers(wrapper, critic)
    actor_params = [
        parameter
        for group in optimizers["actor"].param_groups
        for parameter in group["params"]
    ]
    assert len(actor_params) == len({id(p) for p in actor_params})
    trainable = {id(p) for p in wrapper.parameters() if p.requires_grad}
    assert {id(p) for p in actor_params} == trainable
    critic_probe_ids = {id(p) for p in wrapper.backbone.critic_probe.parameters()}
    assert critic_probe_ids
    assert critic_probe_ids.isdisjoint({id(p) for p in actor_params})
    # Three groups: trunk, combiner, renderer probe. Every actor component
    # uses the same general learning rate as the critic.
    lrs = [group["lr"] for group in optimizers["actor"].param_groups]
    assert lrs == [1e-3] * 3
    assert optimizers["critic"].param_groups[0]["lr"] == 1e-3
    combiner_group = optimizers["actor"].param_groups[1]["params"]
    assert {id(p) for p in combiner_group} == {
        id(p) for p in wrapper.combiner.parameters()
    }
    renderer_group = optimizers["actor"].param_groups[2]["params"]
    assert {id(p) for p in renderer_group} == {
        id(p) for p in wrapper.backbone.policy_probe.parameters()
    }


def test_post_update_drift_measures_the_deployed_policy_move():
    wrapper = _wrapper()
    critic = _critic()
    batch = _rollout(wrapper, batch=4, prompt=5, new_tokens=3)
    refresh_old_statistics(wrapper, critic, batch)
    diagnostic_replay_calls = 0

    def diagnostic_replay(*args, **kwargs):
        nonlocal diagnostic_replay_calls
        diagnostic_replay_calls += 1
        assert not torch.is_grad_enabled()
        return replay_head_inputs(*args, **kwargs)

    unchanged = measure_post_update_policy_drift(
        wrapper,
        [batch],
        replay_max_trajectories=32,
        replay_attention_budget=4 * 1024 * 1024,
        replay_bucket=1,
        replay_function=diagnostic_replay,
    )
    assert max(abs(value) for value in unchanged.values()) < 1e-7

    with torch.no_grad():
        renderer_output = wrapper.backbone.policy_probe.output.weight
        renderer_output.add_(
            torch.linspace(
                -0.02, 0.02, renderer_output.numel()
            ).reshape_as(renderer_output)
        )

    drift = measure_post_update_policy_drift(
        wrapper,
        [batch],
        replay_max_trajectories=32,
        replay_attention_budget=4 * 1024 * 1024,
        replay_bucket=1,
        replay_function=diagnostic_replay,
    )

    assert diagnostic_replay_calls == 2
    assert drift["kl/post_update_token_behavior"] > 0.0
    assert drift["ratio/post_update_token_abs_log_max"] > 0.0


def test_actor_accumulation_defers_the_trunk_step_to_the_caller():
    # The trainer takes one accumulated actor step per PPO epoch: minibatch
    # calls run with actor_step=False (grads accumulate, actor params
    # untouched) and the caller steps once at epoch end.
    torch.manual_seed(41)
    wrapper = _wrapper()
    critic = _critic()
    with torch.no_grad():
        # A zero-init renderer output blocks the policy gradient's only path
        # into the trunk at step 0; de-zero it so the banked gradient is real.
        wrapper.backbone.policy_probe.output.weight.normal_(std=0.02)
    optimizers = _optimizers(wrapper, critic)
    batch = _rollout(wrapper, batch=2, prompt=5, new_tokens=3)
    assign_terminal_rewards(batch, torch.rand(2))
    refresh_old_statistics(wrapper, critic, batch)
    trunk_weight = wrapper.backbone.blocks[0].attn.proj.weight
    before = trunk_weight.detach().clone()
    optimizers["actor"].zero_grad(set_to_none=True)
    metrics = update_minibatch(
        wrapper, critic, batch, optimizers, actor_step=False
    )
    # The actor did not step, but its gradient is banked for the caller.
    torch.testing.assert_close(trunk_weight.detach(), before)
    assert metrics["trunk_grad_norm"] > 0.0
    assert trunk_weight.grad is not None
    optimizers["actor"].step()
    assert not torch.equal(trunk_weight.detach(), before)


def test_critic_accumulation_matches_one_full_effective_minibatch_step():
    torch.manual_seed(43)
    wrapper = _wrapper()
    base_critic = _critic()
    batch = _rollout(wrapper, batch=4, prompt=5, new_tokens=3)
    assign_terminal_rewards(batch, torch.tensor([0.1, 0.4, 0.7, 1.0]))

    full_critic = copy.deepcopy(base_critic)
    accumulated_critic = copy.deepcopy(base_critic)
    full_optimizers = _optimizers(wrapper, full_critic)
    accumulated_optimizers = _optimizers(wrapper, accumulated_critic)
    full_optimizers["critic"].zero_grad(set_to_none=True)
    full_metrics = update_minibatch(
        wrapper,
        full_critic,
        copy.deepcopy(batch),
        full_optimizers,
        value_only=True,
        critic_step=False,
    )

    groups = [
        select_trajectory_rows(batch, rows, batch.stream_length)
        for rows in (torch.tensor([0]), torch.tensor([1, 2, 3]))
    ]
    denominator = batch.action_mask.sum()
    accumulated_optimizers["critic"].zero_grad(set_to_none=True)
    before = [
        parameter.detach().clone()
        for parameter in accumulated_critic.parameters()
    ]
    group_metrics = [
        update_minibatch(
            wrapper,
            accumulated_critic,
            group,
            accumulated_optimizers,
            value_only=True,
            critic_step=False,
            value_action_denominator=denominator,
        )
        for group in groups
    ]
    assert all(
        torch.equal(parameter, initial)
        for parameter, initial in zip(
            accumulated_critic.parameters(), before, strict=True
        )
    )
    for accumulated, full in zip(
        accumulated_critic.parameters(), full_critic.parameters(), strict=True
    ):
        if full.grad is None:
            assert accumulated.grad is None
        else:
            torch.testing.assert_close(
                accumulated.grad, full.grad, rtol=2e-5, atol=2e-6
            )
    full_optimizers["critic"].step()
    accumulated_optimizers["critic"].step()

    accumulated_loss = sum(
        metric["value_loss"] * metric["action_count"]
        for metric in group_metrics
    ) / sum(metric["action_count"] for metric in group_metrics)
    assert accumulated_loss == pytest.approx(full_metrics["value_loss"], rel=1e-6)
    for accumulated, full in zip(
        accumulated_critic.parameters(), full_critic.parameters(), strict=True
    ):
        torch.testing.assert_close(accumulated, full, rtol=3e-4, atol=2e-5)
    for optimizer in (full_optimizers["critic"], accumulated_optimizers["critic"]):
        assert {
            int(state["step"])
            for state in optimizer.state.values()
            if "step" in state
        } == {1}


def test_rollout_replay_and_update_run_under_the_bf16_load_policy():
    # Regression test for the adapter dtype crash: the fp32 adapter must
    # never see a bf16 operand in either the stepwise rollout or the
    # parallel replay, and one full update must stay finite.
    wrapper = _bf16_wrapper()
    critic = _critic()  # the separate critic always runs fp32
    assert wrapper.backbone.tok_emb.weight.dtype == torch.bfloat16
    # Zero-init probe output and a fresh identity combiner would make the
    # clip-fraction assertions below vacuous (both code paths output exactly
    # zero); randomize/enable them so the ratio-one property is load-bearing.
    with torch.no_grad():
        wrapper.backbone.policy_probe.output.weight.normal_(std=0.02)
        wrapper.combiner.carry.weight.normal_(std=0.02)
    batch = _rollout(wrapper, batch=2, prompt=5, new_tokens=3)
    # The carried hidden is stored fp32 even under a bf16 backbone: the
    # combiner computes its injection in fp32 and casts once, so storage
    # must not round the belief first.
    assert batch.hiddens.dtype == torch.float32
    refresh_old_statistics(wrapper, critic, batch)
    assign_terminal_rewards(batch, torch.rand(2))
    metrics = update_minibatch(
        wrapper, critic, batch,
        _optimizers(wrapper, critic, learning_rate=1e-4),
    )
    assert all(
        torch.isfinite(torch.tensor(value)) for value in metrics.values()
    ), metrics
    # With refreshed old statistics, behavior-age-0 ratios start at exactly
    # one, so nothing clips on the first update.
    assert metrics["policy_clip_fraction"] == 0.0
    assert metrics["token_behavior_kl"] == 0.0
    assert metrics["policy_behavior_kl_per_action"] == 0.0
    assert metrics["token_abs_log_ratio_max"] == 0.0
    assert metrics["harmful_positive_log_ratio_max"] == 0.0


def test_later_disjoint_minibatch_keeps_the_pool_behavior_policy_fixed():
    wrapper = _wrapper()
    critic = _critic()
    with torch.no_grad():
        wrapper.backbone.policy_probe.output.weight.normal_(std=0.02)
        wrapper.combiner.carry.weight.normal_(std=0.02)
    first = _rollout(wrapper, batch=4, prompt=5, new_tokens=3, seed=17)
    later = _rollout(wrapper, batch=4, prompt=5, new_tokens=3, seed=19)
    for batch in (first, later):
        assign_terminal_rewards(batch, torch.tensor([1.0, 0.0, 1.0, 0.0]))
        refresh_old_statistics(wrapper, critic, batch)
    frozen_later = {
        name: getattr(later, name).clone()
        for name in (
            "old_values",
            "old_token_logprobs",
        )
    }

    optimizers = _optimizers(wrapper, critic, learning_rate=1e-2)
    first_metrics = update_minibatch(
        wrapper,
        critic,
        first,
        optimizers,
    )
    assert first_metrics["token_abs_log_ratio_max"] == 0.0
    for name, behavior_tensor in frozen_later.items():
        assert torch.equal(getattr(later, name), behavior_tensor)

    # Evaluate MB2 after MB1's optimizer step without refreshing. Its current
    # logprobs must now differ from the still-frozen pre-pool behavior stats.
    later_metrics = update_minibatch(
        wrapper,
        critic,
        later,
        optimizers,
        actor_step=False,
        critic_step=False,
    )
    assert later_metrics["token_abs_log_ratio_max"] > 0.0
    for name, behavior_tensor in frozen_later.items():
        assert torch.equal(getattr(later, name), behavior_tensor)


def test_lambda_one_value_targets_equal_the_terminal_reward_everywhere():
    wrapper = _wrapper()
    batch = _rollout(wrapper, batch=3)
    scores = torch.tensor([0.2, 0.6, 0.9])
    assign_terminal_rewards(batch, scores)
    counts = batch.action_mask.sum(1)
    _, targets = generalized_advantage_estimate(
        batch.rewards, batch.old_values, batch.action_mask, torch.ones_like(counts)
    )
    for row in range(3):
        mask = batch.action_mask[row].bool()
        torch.testing.assert_close(
            targets[row][mask],
            torch.full((int(mask.sum()),), float(scores[row])),
        )


def test_sample_prompt_batch_groups_and_aligns_references():
    seq_len = 16
    rows = torch.arange(3 * seq_len).reshape(3, seq_len)

    class _Loader:
        def next_batch(self, batch_tokens, length, grad_accum):
            assert batch_tokens == 3 * seq_len * grad_accum
            assert length == seq_len
            return rows, None

    prompt_ids, reference_ids = sample_prompt_batch(
        _Loader(), 6, 4, prompts=3, samples_per_prompt=2, seq_len=seq_len
    )
    assert prompt_ids.shape == (6, 6)
    assert reference_ids.shape == (6, 4)
    # repeat_interleave keeps each prompt's samples adjacent, matching the
    # trainer's reshape(-1, samples_per_prompt) group-std diagnostic.
    for group in range(3):
        assert torch.equal(prompt_ids[2 * group], prompt_ids[2 * group + 1])
        assert torch.equal(prompt_ids[2 * group], rows[group, :6])
        assert torch.equal(reference_ids[2 * group], rows[group, 6:10])
    assert not torch.equal(prompt_ids[0], prompt_ids[2])


def test_evaluate_aime_latent_scores_through_the_gate_policy(monkeypatch):
    wrapper = _wrapper()

    class _Tokenizer:
        def eos_id(self) -> int:
            return 5

        def bos_id(self) -> int:
            return -1

        def encode(self, text: str) -> list[int]:
            return [1, 2, 3]

        def decode(self, ids: list[int]) -> str:
            return "Answer: 42" if ids else ""

    rows = [
        {"prompt": [{"content": "question"}], "reward_model": {"ground_truth": "42"}},
        {"prompt": [{"content": "other"}], "reward_model": {"ground_truth": "7"}},
    ]
    import postraining.latent_eval as evaluator

    # Isolate counting/verification from random rollout termination while
    # satisfying the production scorer's explicit-termination contract.
    monkeypatch.setattr(
        evaluator,
        "emitted_token_rows",
        lambda batch: [[5] for _ in range(batch.kind.size(0))],
    )
    metrics = evaluate_aime_latent(
        wrapper, _Tokenizer(), rows, samples=4, max_new_tokens=3,
        max_stream_steps=12, chunk=3, seed=5, device=torch.device("cpu"),
        prompt_tokens=8,
    )
    # Every decode reads "Answer: 42": row one is always right, row two
    # always wrong, so accuracy pins both counting and verification.
    assert metrics["samples"] == 8
    assert metrics["accuracy"] == 0.5
    assert metrics["policy_accuracy"] == 0.5
    assert metrics["prompt_groups"] == 2
    assert metrics["prompt_any_correct_fraction"] == 0.5
    assert metrics["prompt_mixed_reward_fraction"] == 0.0
    assert metrics["prompt_all_correct_fraction"] == 0.5
    assert metrics["prompt_zero_correct_fraction"] == 0.5
    assert metrics["within_group_reward_std"] == 0.0
    assert metrics["ended_fraction"] == 1.0
    assert metrics["emitted_tokens_mean"] == 1.0
    assert metrics["emitted_tokens_p95"] == 1
    assert metrics["emitted_tokens_max"] == 1
    assert (
        metrics["stream_actions_max"]
        >= metrics["stream_actions_p95"]
        >= metrics["stream_actions_mean"]
        > 0
    )
    assert metrics["recurrent_steps_per_rollout_max"] > 0
    # The eval must not perturb training RNG state.
    before = torch.get_rng_state()
    evaluate_aime_latent(
        wrapper, _Tokenizer(), rows[:1], samples=2, max_new_tokens=2,
        max_stream_steps=8, chunk=2, seed=5, device=torch.device("cpu"),
        prompt_tokens=8,
    )
    assert torch.equal(before, torch.get_rng_state())


def test_latent_eval_headline_scores_every_native_policy_attempt(monkeypatch):
    wrapper = _wrapper()

    class _Tokenizer:
        def eos_id(self):
            return 5

        def bos_id(self):
            return -1

        def encode(self, _text):
            return [1, 2, 3]

        def decode(self, ids):
            return "Answer: 42" if 1 in ids else "Answer: 7"

    import postraining.latent_eval as evaluator

    def alternating_rows(batch):
        return [
            [1 if row % 2 == 0 else 2, 5]
            for row in range(batch.kind.size(0))
        ]

    monkeypatch.setattr(
        evaluator, "emitted_token_rows", alternating_rows
    )
    metrics = evaluate_aime_latent(
        wrapper,
        _Tokenizer(),
        [{"prompt": [{"content": "q"}], "reward_model": {"ground_truth": "42"}}],
        samples=4,
        max_new_tokens=3,
        max_stream_steps=12,
        chunk=4,
        seed=7,
        device=torch.device("cpu"),
        prompt_tokens=8,
    )

    assert metrics["policy_samples"] == 4
    assert metrics["policy_accuracy"] == 0.5
    assert metrics["accuracy"] == 0.5


def test_evaluate_aime_latent_captures_first_four_problems_in_dataset_order(
    monkeypatch,
):
    wrapper = _wrapper()

    class _Tokenizer:
        def eos_id(self) -> int:
            return 5

        def bos_id(self) -> int:
            return -1

        def encode(self, text: str) -> list[int]:
            # Deliberately reverse length versus dataset order to exercise
            # evaluation's stable length bucketing.
            return list(range(1, len(text) + 1))

        def decode(self, ids: list[int]) -> str:
            return "work\nAnswer: 42"

        def id_to_piece(self, token_id: int) -> str:
            return "<|endoftext|>"

    rows = [
        {
            "prompt": [{"content": text}],
            "reward_model": {"ground_truth": "42"},
            "extra_info": {"index": 100 + index},
        }
        for index, text in enumerate(("longest", "x", "medium", "xx", "ignored"))
    ]
    import postraining.latent_eval as evaluator

    monkeypatch.setattr(
        evaluator,
        "emitted_token_rows",
        lambda batch: [[5] for _ in range(batch.kind.size(0))],
    )
    attempts: list[dict[str, object]] = []
    metrics = evaluate_aime_latent(
        wrapper,
        _Tokenizer(),
        rows,
        samples=4,
        max_new_tokens=2,
        max_stream_steps=8,
        chunk=4,
        seed=5,
        device=torch.device("cpu"),
        prompt_tokens=16,
        batch_trajectories=20,
        captured_attempts=attempts,
    )

    assert metrics["samples"] == 20
    assert len(attempts) == 16
    assert [
        (attempt["problem_index"], attempt["sample_index"])
        for attempt in attempts
    ] == [(problem, sample) for problem in range(4) for sample in range(4)]
    assert [attempts[problem * 4]["prompt"] for problem in range(4)] == [
        "longest",
        "x",
        "medium",
        "xx",
    ]
    assert [attempts[problem * 4]["dataset_index"] for problem in range(4)] == [
        "100",
        "101",
        "102",
        "103",
    ]
    for attempt in attempts:
        assert attempt["terminated"] is True
        assert attempt["correct"] is True
        assert attempt["parsed_answer"] == "42"
        assert attempt["emitted_token_ids"] == [5]
        assert attempt["emitted_token_count"] == 1
        # The lone emitted token is the stub's EOS, so the display
        # segments reduce to its terminal marker (no prefix ids here).
        assert attempt["emitted_segments"] == [
            {
                "kind": "special",
                "role": "eos",
                "text": "<|endoftext|>",
                "token_id": 5,
                "source": "text",
            }
        ]

    full_capture: list[dict[str, object]] = []
    evaluate_aime_latent(
        wrapper,
        _Tokenizer(),
        rows,
        samples=4,
        max_new_tokens=2,
        max_stream_steps=8,
        chunk=3,
        seed=5,
        device=torch.device("cpu"),
        prompt_tokens=16,
        batch_trajectories=12,
        captured_attempts=full_capture,
        capture_problem_count=5,
        capture_samples_per_problem=4,
    )
    assert [
        (attempt["problem_index"], attempt["sample_index"])
        for attempt in full_capture
    ] == [(problem, sample) for problem in range(5) for sample in range(4)]


def test_evaluate_aime_latent_uses_native_policy_across_chunks(
    monkeypatch,
):
    wrapper = _wrapper()

    class _Tokenizer:
        def eos_id(self) -> int:
            return 5

        def bos_id(self) -> int:
            return -1

        def encode(self, text: str) -> list[int]:
            return [1, 2, 3]

        def decode(self, ids: list[int]) -> str:
            return "Answer: 42"

    import postraining.latent_eval as evaluator

    widths = []
    original = evaluator.rollout_continuations

    def _spy(*args, **kwargs):
        widths.append(kwargs["prompt_repeats"])
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluator, "rollout_continuations", _spy)
    evaluate_aime_latent(
        wrapper,
        _Tokenizer(),
        [{"prompt": [{"content": "q"}], "reward_model": {"ground_truth": "42"}}],
        samples=4,
        max_new_tokens=2,
        max_stream_steps=8,
        chunk=3,
        seed=5,
        device=torch.device("cpu"),
        prompt_tokens=8,
    )
    assert widths == [3, 1]


def test_finished_row_compaction_preserves_original_row_attribution():
    wrapper = _deterministic_wrapper()
    prompt_ids = torch.randint(0, 32, (8, 4))
    def run(compact_finished: bool):
        torch.manual_seed(31)
        return rollout_continuations(
            wrapper,
            prompt_ids,
            max_new_tokens=1,
            max_stream_steps=1,
            temperature=1.0,
            top_p=1e-6,
            compact_finished=compact_finished,
        )

    compact = run(True)
    reference = run(False)
    for field in (
        "kind",
        "token_ids",
        "hiddens",
        "action_mask",
    ):
        torch.testing.assert_close(getattr(compact, field), getattr(reference, field))
    boundary = compact.prompt_length - 1
    assert compact.action_mask[:, boundary].bool().all()


def test_compiled_full_batch_uses_one_fixed_finished_tail_shape(
    monkeypatch,
):
    wrapper = _deterministic_wrapper()
    prompt_ids = torch.randint(0, 32, (128, 4))
    original_step_core = wrapper.step_core
    compiled_batch_sizes: list[int] = []

    def compiled_step_core(next_input, *args, **kwargs):
        compiled_batch_sizes.append(next_input.size(0))
        return original_step_core(next_input, *args, **kwargs)

    wrapper.step_core = compiled_step_core
    calls = 0

    def staged_tokens(logits, _temperature, _top_p, **_kwargs):
        nonlocal calls
        tokens = torch.full(
            (logits.size(0),), 6, dtype=torch.long, device=logits.device
        )
        if calls == 0:
            # Leave 12 live rows so the fixed B16 tail has to retain four
            # inert fillers, not merely compact to exactly 16 survivors.
            tokens[:116] = 5
        calls += 1
        return tokens

    import postraining.latent_rollout as latent_rollout

    monkeypatch.setattr(latent_rollout, "top_p_sample", staged_tokens)
    batch = rollout_continuations(
        wrapper,
        prompt_ids,
        max_new_tokens=32,
        max_stream_steps=33,
        temperature=1.0,
        top_p=0.7,
        stop_ids=(5,),
        compact_finished=True,
        finished_batch_size=16,
        record_likelihoods=False,
    )

    emitted = emitted_token_rows(batch)
    assert emitted[:116] == [[5]] * 116
    assert all(row == [6] * 32 for row in emitted[116:])
    assert compiled_batch_sizes
    assert set(compiled_batch_sizes) == {16, 128}
    assert wrapper.step_core is compiled_step_core


def test_fixed_finished_tail_never_expands_a_smaller_batch(monkeypatch):
    wrapper = _deterministic_wrapper()
    prompt_ids = torch.randint(0, 32, (8, 4))
    original_step_core = wrapper.step_core
    observed_batch_sizes: list[int] = []

    def observed_step_core(next_input, *args, **kwargs):
        observed_batch_sizes.append(next_input.size(0))
        return original_step_core(next_input, *args, **kwargs)

    wrapper.step_core = observed_step_core
    calls = 0

    def staged_tokens(logits, _temperature, _top_p, **_kwargs):
        nonlocal calls
        tokens = torch.full(
            (logits.size(0),), 6, dtype=torch.long, device=logits.device
        )
        if calls == 1:
            tokens[:6] = 5
        calls += 1
        return tokens

    import postraining.latent_rollout as latent_rollout

    monkeypatch.setattr(latent_rollout, "top_p_sample", staged_tokens)
    rollout_continuations(
        wrapper,
        prompt_ids,
        max_new_tokens=32,
        max_stream_steps=33,
        temperature=1.0,
        top_p=0.7,
        stop_ids=(5,),
        compact_finished=True,
        finished_batch_size=16,
        record_likelihoods=False,
    )
    assert set(observed_batch_sizes) == {8}


def test_progressive_compaction_shrinks_above_the_fixed_tail(monkeypatch):
    # With a fixed finished tail configured, a wave of early finishers that
    # still leaves MORE survivors than the tail width must shrink the batch
    # progressively under the >=25%-dead rule instead of stepping the dead
    # rows at full width until the final tail snap.
    wrapper = _deterministic_wrapper()
    prompt_ids = torch.randint(0, 32, (128, 4))
    original_step_core = wrapper.step_core
    observed_batch_sizes: list[int] = []

    def observed_step_core(next_input, *args, **kwargs):
        observed_batch_sizes.append(next_input.size(0))
        return original_step_core(next_input, *args, **kwargs)

    wrapper.step_core = observed_step_core
    calls = 0

    def staged_tokens(logits, _temperature, _top_p, **_kwargs):
        nonlocal calls
        tokens = torch.full(
            (logits.size(0),), 6, dtype=torch.long, device=logits.device
        )
        if calls == 0:
            # 40 dead of 128 (>=25%, 88 survivors > 16): progressive shrink.
            tokens[:40] = 5
        elif calls == 16:
            # Batch order is now the 88 survivors; leave 12 for the tail.
            tokens[: logits.size(0) - 12] = 5
        calls += 1
        return tokens

    import postraining.latent_rollout as latent_rollout

    monkeypatch.setattr(latent_rollout, "top_p_sample", staged_tokens)
    batch = rollout_continuations(
        wrapper,
        prompt_ids,
        max_new_tokens=40,
        max_stream_steps=48,
        temperature=1.0,
        top_p=0.7,
        stop_ids=(5,),
        compact_finished=True,
        finished_batch_size=16,
        record_likelihoods=False,
    )

    emitted = emitted_token_rows(batch)
    assert emitted[:40] == [[5]] * 40
    assert emitted[40:116] == [[6] * 16 + [5]] * 76
    assert all(row == [6] * 40 for row in emitted[116:])
    # Full width, one intermediate width, then the fixed tail — never an
    # intermediate width equal to the tail (which would engage a tail graph
    # early), never an expansion.
    assert sorted(set(observed_batch_sizes)) == [16, 88, 128]


def test_finished_row_compaction_preserves_model_dependent_survivor_tokens(
    monkeypatch,
):
    wrapper = _deterministic_wrapper()
    prompt_ids = torch.randint(1, 32, (128, 4))
    prompt_ids[:64, 0] = 0
    prompt_lengths = torch.tensor([3] * 64 + [4] * 64)
    calls = 0

    def terminate_then_argmax(logits, _temperature, _top_p, **_kwargs):
        nonlocal calls
        if calls == 1:
            tokens = logits.argmax(-1)
            tokens[:80] = 5
        else:
            tokens = logits.argmax(-1)
        calls += 1
        return tokens

    import postraining.latent_rollout as latent_rollout

    monkeypatch.setattr(
        latent_rollout, "top_p_sample", terminate_then_argmax
    )

    def run(compact_finished: bool):
        nonlocal calls
        calls = 0
        return rollout_continuations(
            wrapper,
            prompt_ids,
            max_new_tokens=32,
            max_stream_steps=33,
            temperature=1.0,
            top_p=0.7,
            stop_ids=(5,),
            prompt_lengths=prompt_lengths,
            compact_finished=compact_finished,
            record_likelihoods=False,
        )

    fixed = emitted_token_rows(run(False))
    compacted = emitted_token_rows(run(True))
    assert all(row[-1] == 5 and len(row) == 2 for row in compacted[:80])
    assert compacted[:80] == fixed[:80]
    assert compacted[80:] == fixed[80:]


def test_static_tail_switch_matches_the_dynamic_tail(monkeypatch):
    """The tail_caches switch is invisible to the sampled trajectories.

    Mixed prompt lengths make left-pad masking load-bearing: if the
    fixed-width tail mask wrongly attended a pad slot's garbage K/V, the
    argmax tokens below would flip. The third run reuses the DIRTY caches
    from the second to pin that stale slots beyond the live prefix stay
    masked, and spies on ``tail_step_core`` to pin the constant shapes the
    CUDA graph relies on.
    """
    wrapper = _deterministic_wrapper()
    generator = torch.Generator().manual_seed(17)
    prompt_ids = torch.randint(1, 32, (64, 4), generator=generator)
    prompt_ids[:32, 0] = 0
    prompt_lengths = torch.tensor([3] * 32 + [4] * 32)
    calls = 0

    def terminate_then_argmax(logits, _temperature, _top_p, **_kwargs):
        nonlocal calls
        tokens = logits.argmax(-1)
        if calls == 1:
            # Leave 12 live rows so the fixed B16 tail retains 4 fillers —
            # six of them LEFT-PADDED (rows 0-5) so a tail mask that wrongly
            # attends a surviving row's pad slot would flip its argmax.
            tokens[6:58] = 5
        calls += 1
        return tokens

    import postraining.latent_rollout as latent_rollout

    monkeypatch.setattr(latent_rollout, "top_p_sample", terminate_then_argmax)

    seen_shapes = set()
    original_step_core = wrapper.step_core

    def spy_tail_core(next_input, caches, position, key_mask, block_mask=None):
        seen_shapes.add((next_input.size(0), tuple(key_mask.shape)))
        return original_step_core(
            next_input, caches, position, key_mask, block_mask
        )

    def run(tail_caches=None, tail_step_core=None):
        nonlocal calls
        calls = 0
        return rollout_continuations(
            wrapper,
            prompt_ids,
            max_new_tokens=32,
            max_stream_steps=33,
            temperature=1.0,
            top_p=0.7,
            stop_ids=(5,),
            prompt_lengths=prompt_lengths,
            tensor_positions=True,
            compact_finished=True,
            finished_batch_size=16,
            record_likelihoods=False,
            tail_caches=tail_caches,
            tail_step_core=tail_step_core,
        )

    dynamic = emitted_token_rows(run())
    static_caches = wrapper.make_static_generation_cache(
        16, 40, torch.device("cpu")
    )
    static = emitted_token_rows(run(tail_caches=static_caches))
    assert static == dynamic
    # The switch actually landed the survivors in the static caches.
    assert sum(float(cache[0].abs().sum()) for cache in static_caches) > 0.0

    reused = emitted_token_rows(
        run(tail_caches=static_caches, tail_step_core=spy_tail_core)
    )
    assert reused == static
    assert seen_shapes == {(16, (16, 40))}


def _deterministic_nano_wrapper() -> LatentThoughtModel:
    """``_deterministic_wrapper`` on the RoPE nano trunk.

    The flex decode path exists for one QK product over a K/V cache pair,
    which is the nano trunk; PoPE scores a complex inner product over a
    k_real/k_imag pair and keeps the boolean path.
    """
    torch.manual_seed(3)
    backbone = NanoGPTBackbone(vocab_size=32, num_layers=2, model_dim=256)
    backbone = backbone.float().eval()
    with torch.no_grad():
        backbone.proj.weight.normal_(std=0.05)
        backbone.proj.bias.normal_(std=0.05)
    return LatentThoughtModel(backbone)


def test_flex_decode_mask_reproduces_the_boolean_row_mask(monkeypatch):
    """The block table IS the boolean tail mask, not an approximation of it.

    Mixed prompt lengths make left-pad masking load-bearing, and the flex path
    additionally attends the whole static tail cache rather than a narrowed
    prefix: a block table one slot too wide would let a pad slot or a slot past
    the write head into the softmax, and the argmax tokens would flip.
    Compaction runs on both sides, so the run reaches the tail switch through
    the same surviving-row bookkeeping either way.
    """
    wrapper = _deterministic_nano_wrapper()
    generator = torch.Generator().manual_seed(17)
    prompt_ids = torch.randint(1, 32, (64, 4), generator=generator)
    prompt_ids[:32, 0] = 0
    prompt_lengths = torch.tensor([3] * 32 + [4] * 32)
    calls = 0

    def terminate_then_argmax(logits, _temperature, _top_p, **_kwargs):
        nonlocal calls
        tokens = logits.argmax(-1)
        # Nobody stops by accident: the 12 survivors must reach the first
        # 16-step synchronization point for the fixed B16 tail to engage.
        tokens[tokens == 5] = 6
        if calls == 1:
            tokens[6:58] = 5
        calls += 1
        return tokens

    import postraining.latent_rollout as latent_rollout

    monkeypatch.setattr(latent_rollout, "top_p_sample", terminate_then_argmax)
    cpu = torch.device("cpu")

    def run(**overrides):
        nonlocal calls
        calls = 0
        kwargs = dict(
            max_new_tokens=32,
            max_stream_steps=33,
            temperature=1.0,
            top_p=0.7,
            stop_ids=(5,),
            prompt_lengths=prompt_lengths,
            tensor_positions=True,
            compact_finished=True,
            finished_batch_size=16,
            record_likelihoods=False,
        )
        kwargs.update(overrides)
        return rollout_continuations(wrapper, prompt_ids, **kwargs)

    # 37 slots of stream rounded up to whole 16-wide flex KV blocks.
    boolean_caches = wrapper.make_static_generation_cache(16, 48, cpu)
    boolean = emitted_token_rows(run(tail_caches=boolean_caches))
    flex_caches = wrapper.make_static_generation_cache(16, 48, cpu)
    flex = emitted_token_rows(
        run(
            tail_caches=flex_caches,
            tail_decode_mask=DecodeRangeMask(16, 48, cpu, block_size=16),
        )
    )
    assert flex == boolean
    # The switch actually landed the survivors in the static caches, so the
    # agreement above is the tail's doing and not a run that never reached it.
    assert sum(float(cache[0].abs().sum()) for cache in flex_caches) > 0.0


def test_flex_decode_main_loop_and_tail_compose(monkeypatch):
    """The production combination: bucketed main loop handing off to the tail.

    The two masks are covered separately above, and each is sound alone. What
    neither covers is the HANDOFF, which is where their differing widths meet:
    the main loop rolls on a per-chunk cache and the tail on a wider static
    one, and the switch copies only ``[0, live_prefix)`` between them. If the
    main loop's bucket left a filler row in the survivor set, or the copy
    landed the survivors at the wrong rows, the tail would score a different
    trajectory and the emitted tokens would diverge from the boolean run.
    """
    wrapper = _deterministic_nano_wrapper()
    generator = torch.Generator().manual_seed(17)
    prompt_ids = torch.randint(1, 32, (64, 4), generator=generator)
    prompt_ids[:32, 0] = 0
    prompt_lengths = torch.tensor([3] * 32 + [4] * 32)
    calls = 0

    def terminate_then_argmax(logits, _temperature, _top_p, **_kwargs):
        nonlocal calls
        tokens = logits.argmax(-1)
        tokens[tokens == 5] = 6
        if calls == 1:
            tokens[6:58] = 5
        calls += 1
        return tokens

    import postraining.latent_rollout as latent_rollout

    monkeypatch.setattr(latent_rollout, "top_p_sample", terminate_then_argmax)
    cpu = torch.device("cpu")

    def run(**overrides):
        nonlocal calls
        calls = 0
        kwargs = dict(
            max_new_tokens=32,
            max_stream_steps=33,
            temperature=1.0,
            top_p=0.7,
            stop_ids=(5,),
            prompt_lengths=prompt_lengths,
            tensor_positions=True,
            compact_finished=True,
            finished_batch_size=16,
            record_likelihoods=False,
        )
        kwargs.update(overrides)
        return rollout_continuations(wrapper, prompt_ids, **kwargs)

    boolean_caches = wrapper.make_static_generation_cache(16, 48, cpu)
    boolean = emitted_token_rows(run(tail_caches=boolean_caches))

    flex_caches = wrapper.make_static_generation_cache(16, 48, cpu)
    flex = emitted_token_rows(
        run(
            tail_caches=flex_caches,
            # A bucket well below the 64 rows rolled out, so the main loop
            # actually compacts through intermediate widths before the snap
            # instead of jumping straight to the tail.
            decode_mask=DecodeRangeMask(
                64, 48, cpu, block_size=16, row_bucket=8
            ),
            tail_decode_mask=DecodeRangeMask(16, 48, cpu, block_size=16),
        )
    )
    assert flex == boolean
    assert sum(float(cache[0].abs().sum()) for cache in flex_caches) > 0.0


def test_graph_arena_rollout_matches_the_allocating_rollout(monkeypatch):
    """A caller-owned decode arena must change nothing but where KV lives.

    This is the CUDA-graph configuration: one static arena for the whole run
    instead of a cache sized per chunk, therefore no compaction and one fixed
    row count. Two things could silently diverge. The prompt fan-out now
    expands into the caller's tensors rather than a fresh allocation, so a
    wrong target leaves samples sharing or missing prefix KV. And the arena is
    REUSED across rollouts, so the second run starts on the first run's KV --
    unreachable only because each row's range starts at its own left pad and
    stops at the write head. Running the same prompts twice through one arena
    and demanding both match the allocating path pins exactly that.
    """
    wrapper = _deterministic_nano_wrapper()
    generator = torch.Generator().manual_seed(29)
    prompt_ids = torch.randint(1, 32, (8, 4), generator=generator)
    prompt_ids[:4, 0] = 0
    prompt_lengths = torch.tensor([3] * 4 + [4] * 4)
    cpu = torch.device("cpu")
    repeats = 3
    rows, width = prompt_ids.size(0) * repeats, 48

    def argmax(logits, _temperature, _top_p, **_kwargs):
        tokens = logits.argmax(-1)
        tokens[tokens == 5] = 6
        return tokens

    import postraining.latent_rollout as latent_rollout

    monkeypatch.setattr(latent_rollout, "top_p_sample", argmax)

    def run(**overrides):
        kwargs = dict(
            max_new_tokens=32,
            max_stream_steps=width - prompt_ids.size(1),
            temperature=1.0,
            top_p=0.7,
            stop_ids=(5,),
            prompt_lengths=prompt_lengths,
            prompt_repeats=repeats,
            tensor_positions=True,
            record_likelihoods=False,
            decode_mask=DecodeRangeMask(
                rows, width, cpu, block_size=16, row_bucket=8
            ),
        )
        kwargs.update(overrides)
        return rollout_continuations(wrapper, prompt_ids, **kwargs)

    allocating = emitted_token_rows(run(compact_finished=True))
    arena = wrapper.make_static_generation_cache(rows, width, cpu)
    first = emitted_token_rows(run(caches=arena, compact_finished=False))
    # Deliberately not re-zeroed: reuse across pools is the whole point of a
    # static arena, and stale KV must already be unreachable.
    second = emitted_token_rows(run(caches=arena, compact_finished=False))

    assert first == allocating
    assert second == allocating
    assert sum(float(cache[0].abs().sum()) for cache in arena) > 0.0


def test_a_short_chunk_pads_up_to_the_arena_without_changing_a_row(monkeypatch):
    """Fewer prompts than the arena holds must run, and run unchanged.

    The arena fixes the row count for the whole run, but real chunks are
    routinely short: the value warmup takes fewer prompts than there are
    rollout groups, the final pool takes whatever prompts remain, and
    --consume-all-prompts takes an arbitrary count. Rejecting those (the first
    shape of this flag) kills every fresh run at its first warmup rollout, and
    running them narrower records a second graph at a second shape, which is
    the cost the arena exists to remove. They pad up instead. This pins that
    the padding is invisible: the returned batch holds exactly the real rows,
    and every one is identical to the same chunk on an exactly-sized arena.

    Row-for-row equality holds here because sampling is patched to argmax.
    Under real sampling it would not: every draw runs at the PADDED row count,
    so a short chunk's real rows diverge from the same chunk unpadded after
    the first step. That is a reproducibility property of the flag, not a
    defect -- the arena width is fixed by config -- but it is why this test
    pins the plumbing rather than the samples.
    """
    wrapper = _deterministic_nano_wrapper()
    generator = torch.Generator().manual_seed(31)
    prompt_ids = torch.randint(1, 32, (8, 4), generator=generator)
    prompt_ids[:4, 0] = 0
    prompt_lengths = torch.tensor([3] * 4 + [4] * 4)
    cpu = torch.device("cpu")
    repeats, width = 3, 48
    rows = prompt_ids.size(0) * repeats

    sampled_steps = [0]

    def argmax(logits, _temperature, _top_p, **_kwargs):
        sampled_steps[0] += 1
        tokens = logits.argmax(-1)
        tokens[tokens == 5] = 6
        if sampled_steps[0] > 12:
            # Every REAL row hits the stop token here and no filler ever does.
            # Fillers copy the last prompt but sample their own continuations,
            # so outliving the whole chunk is their ordinary behavior, not a
            # contrivance; a live one would hold the loop open for the rest of
            # the stream.
            tokens[:rows] = 5
        return tokens

    import postraining.latent_rollout as latent_rollout

    monkeypatch.setattr(latent_rollout, "top_p_sample", argmax)

    def run(arena_rows):
        sampled_steps[0] = 0
        return rollout_continuations(
            wrapper,
            prompt_ids,
            max_new_tokens=32,
            max_stream_steps=width - prompt_ids.size(1),
            temperature=1.0,
            top_p=0.7,
            stop_ids=(5,),
            prompt_lengths=prompt_lengths,
            prompt_repeats=repeats,
            tensor_positions=True,
            record_likelihoods=False,
            compact_finished=False,
            caches=wrapper.make_static_generation_cache(arena_rows, width, cpu),
            decode_mask=DecodeRangeMask(
                arena_rows, width, cpu, block_size=16, row_bucket=8
            ),
        )

    exact = run(rows)
    exact_steps = sampled_steps[0]
    # 12 filler rows: four filler prompts, each fanned out to ``repeats``.
    padded = run(rows + 12)

    # Fillers are ended before the first step, so they never hold the loop
    # open past the last real row -- the property that makes their empty key
    # range, and therefore the whole padding scheme, free.
    assert sampled_steps[0] == exact_steps
    for field in fields(exact):
        if field.name == "replay_layout_token":
            continue
        left, right = getattr(exact, field.name), getattr(padded, field.name)
        if not isinstance(left, torch.Tensor):
            assert left == right, field.name
            continue
        assert right.size(0) == rows, field.name
        torch.testing.assert_close(right, left, msg=field.name)


def test_flex_decode_main_loop_matches_the_boolean_rollout(monkeypatch):
    """Bucketing plus empty ranges must not move a single recorded value.

    Two things here are not obviously free. Rounding the survivor count UP
    means the batch carries filler rows that the boolean path would have
    dropped. Giving every STOPPED row an empty range means it now reads zero
    out of attention where before it read a real (discarded) value — and the
    loop keeps stepping those rows, so their gate decisions and their token
    writes change. That is only safe if every consumer masks them. Compare the
    whole batch, not just the emitted text, so an unmasked leak shows up.
    """
    wrapper = _deterministic_nano_wrapper()
    generator = torch.Generator().manual_seed(23)
    prompt_ids = torch.randint(1, 32, (24, 4), generator=generator)
    prompt_ids[:12, 0] = 0
    prompt_lengths = torch.tensor([3] * 12 + [4] * 12)
    calls = 0

    def argmax(logits, _temperature, _top_p, **_kwargs):
        # Deliberately NOT indexed by row position: the two arms compact to
        # different widths, so their physical row order differs and any
        # position-indexed rule would stop different sequences in each. Letting
        # the stop token fall out of the logits keeps the decision a property
        # of the sequence, which is what makes the arms comparable at all.
        nonlocal calls
        calls += 1
        return logits.argmax(-1)

    import postraining.latent_rollout as latent_rollout

    monkeypatch.setattr(latent_rollout, "top_p_sample", argmax)
    cpu = torch.device("cpu")

    def run(**overrides):
        nonlocal calls
        calls = 0
        kwargs = dict(
            max_new_tokens=24,
            max_stream_steps=25,
            temperature=1.0,
            top_p=0.7,
            stop_ids=(5,),
            prompt_lengths=prompt_lengths,
            tensor_positions=True,
            compact_finished=True,
            sync_every=2,
            record_likelihoods=False,
        )
        kwargs.update(overrides)
        return rollout_continuations(wrapper, prompt_ids, **kwargs)

    boolean = run()
    # 29 slots of stream rounded up to whole 16-wide flex KV blocks.
    flex = run(decode_mask=DecodeRangeMask(24, 32, cpu, block_size=16))

    assert emitted_token_rows(flex) == emitted_token_rows(boolean)
    mask = boolean.action_mask.bool()
    assert torch.equal(flex.action_mask, boolean.action_mask)
    assert torch.equal(flex.kind[mask], boolean.kind[mask])
    assert torch.equal(flex.token_ids[mask], boolean.token_ids[mask])
    assert torch.isfinite(flex.hiddens).all()


def test_flex_decode_bucket_rounds_the_survivor_count_up(monkeypatch):
    """The bucket is what makes the shapes static; pin it directly.

    Compaction that lands on an exact survivor count would respecialize the
    compiled step per count, which is the whole reason the main loop could not
    take flex decoding before.
    """
    wrapper = _deterministic_nano_wrapper()
    # Deliberately not a multiple of the bucket: the FIRST width is whatever
    # the caller rolls out, and only the compacted widths are bucketed.
    generator = torch.Generator().manual_seed(23)
    prompt_ids = torch.randint(1, 32, (24, 4), generator=generator)
    prompt_ids[:12, 0] = 0
    import postraining.latent_rollout as latent_rollout

    monkeypatch.setattr(
        latent_rollout,
        "top_p_sample",
        lambda logits, _t, _p, **_k: logits.argmax(-1),
    )
    cpu = torch.device("cpu")
    widths: list[int] = []
    original = wrapper.step

    def record(input_latent, caches, position, key_mask=None, block_mask=None):
        widths.append(caches[0][0].size(0))
        return original(input_latent, caches, position, key_mask, block_mask)

    wrapper.step = record
    try:
        rollout_continuations(
            wrapper,
            prompt_ids,
            max_new_tokens=24,
            max_stream_steps=25,
            temperature=1.0,
            top_p=0.7,
            stop_ids=(5,),
            prompt_lengths=torch.tensor([3] * 12 + [4] * 12),
            tensor_positions=True,
            compact_finished=True,
            sync_every=2,
            record_likelihoods=False,
            decode_mask=DecodeRangeMask(
                24, 32, cpu, block_size=16, row_bucket=4
            ),
        )
    finally:
        wrapper.step = original
    assert widths, "the rollout never stepped"
    compacted = [width for width in widths if width != 24]
    assert compacted, "the rollout never compacted, so nothing was bucketed"
    for width in compacted:
        assert width % 4 == 0, f"batch {width} is not a multiple of the bucket"


def test_flex_decode_mask_validation_rejects_misfit_masks():
    wrapper = _deterministic_nano_wrapper()
    prompt_ids = torch.randint(1, 32, (8, 4))
    cpu = torch.device("cpu")

    def run(**overrides):
        kwargs = dict(
            max_new_tokens=4,
            max_stream_steps=8,
            temperature=1.0,
            top_p=1.0,
            tensor_positions=True,
            record_likelihoods=False,
        )
        kwargs.update(overrides)
        return rollout_continuations(wrapper, prompt_ids, **kwargs)

    tail_kwargs = dict(
        compact_finished=True,
        finished_batch_size=4,
        tail_caches=wrapper.make_static_generation_cache(4, 16, cpu),
    )
    with pytest.raises(ValueError, match="multiple of the flex KV block"):
        DecodeRangeMask(8, 20, cpu, block_size=16)
    with pytest.raises(ValueError, match="static tail caches"):
        run(tail_decode_mask=DecodeRangeMask(4, 16, cpu, block_size=16))
    with pytest.raises(ValueError, match="match the tail cache width"):
        run(tail_decode_mask=DecodeRangeMask(4, 32, cpu, block_size=16),
            **tail_kwargs)
    with pytest.raises(ValueError, match="fewer than"):
        run(tail_decode_mask=DecodeRangeMask(2, 16, cpu, block_size=16),
            **tail_kwargs)


def test_static_tail_validation_rejects_misfit_caches():
    wrapper = _deterministic_wrapper()
    prompt_ids = torch.randint(1, 32, (8, 4))
    cpu = torch.device("cpu")

    def run(**overrides):
        kwargs = dict(
            max_new_tokens=4,
            max_stream_steps=8,
            temperature=1.0,
            top_p=1.0,
            tensor_positions=True,
            compact_finished=True,
            finished_batch_size=4,
            record_likelihoods=False,
        )
        kwargs.update(overrides)
        return rollout_continuations(wrapper, prompt_ids, **kwargs)

    good = wrapper.make_static_generation_cache(4, 12, cpu)
    with pytest.raises(ValueError, match="finished_batch_size"):
        run(tail_caches=good, finished_batch_size=None)
    with pytest.raises(ValueError, match="tensor_positions"):
        run(tail_caches=good, tensor_positions=False)
    with pytest.raises(ValueError, match="do not fit"):
        run(tail_caches=wrapper.make_static_generation_cache(3, 12, cpu))
    with pytest.raises(ValueError, match="do not fit"):
        run(tail_caches=wrapper.make_static_generation_cache(4, 8, cpu))
    with pytest.raises(ValueError, match="dtype"):
        run(
            tail_caches=wrapper.make_static_generation_cache(
                4, 12, cpu, dtype=torch.bfloat16
            )
        )


def test_evaluate_aime_latent_batches_unequal_prompt_groups_without_replay_storage(
    monkeypatch,
):
    wrapper = _wrapper()

    class _Tokenizer:
        def eos_id(self) -> int:
            return 5

        def bos_id(self) -> int:
            return -1

        def encode(self, text: str) -> list[int]:
            return list(range(1, len(text) + 1))

        def decode(self, ids: list[int]) -> str:
            return "Answer: 42"

    rows = [
        {
            "prompt": [{"content": text}],
            "reward_model": {"ground_truth": "42"},
        }
        for text in ("a", "abcd", "ab")
    ]
    import postraining.latent_eval as evaluator

    seen = []
    original = evaluator.rollout_continuations

    def _spy(wrapper_arg, prompt_ids, *args, **kwargs):
        seen.append(
            (
                prompt_ids.clone(),
                kwargs["prompt_lengths"].clone(),
                kwargs["replay_storage"],
                kwargs["record_likelihoods"],
                args[2],
                args[3],
                kwargs["prompt_repeats"],
            )
        )
        return original(wrapper_arg, prompt_ids, *args, **kwargs)

    monkeypatch.setattr(evaluator, "rollout_continuations", _spy)
    metrics = evaluate_aime_latent(
        wrapper, _Tokenizer(), rows, samples=4, max_new_tokens=2,
        max_stream_steps=8, chunk=4, seed=5, device=torch.device("cpu"),
        prompt_tokens=8, batch_trajectories=12,
        temperature=0.8, top_p=0.6,
    )
    assert metrics["samples"] == 12
    assert len(seen) == 1
    (
        prompt_ids,
        prompt_lengths,
        replay_storage,
        record_likelihoods,
        temperature,
        top_p,
        prompt_repeats,
    ) = seen[0]
    # Unique prompts pass through once; members are expanded structurally
    # through ``prompt_repeats`` after the shared deterministic prefix.
    assert prompt_ids.shape == (3, 4)
    assert prompt_repeats == 4
    assert prompt_lengths.tolist() == [1, 2, 4]
    assert prompt_ids[0].tolist() == [0, 0, 0, 1]
    assert prompt_ids[1].tolist() == [0, 0, 1, 2]
    assert prompt_ids[2].tolist() == [1, 2, 3, 4]
    assert replay_storage is False
    assert record_likelihoods is False
    assert temperature == 0.8
    assert top_p == 0.6


def test_dynamic_compiled_step_accepts_tensor_positions_and_prompt_masks():
    wrapper = _wrapper()
    original = wrapper.step_core
    wrapper.step_core = torch.compile(
        wrapper.step_core, backend="eager", fullgraph=True, dynamic=True
    )
    prompt_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    batch = rollout_continuations(
        wrapper,
        prompt_ids,
        max_new_tokens=2,
        max_stream_steps=8,
        temperature=1.0,
        top_p=1.0,
        prompt_lengths=torch.tensor([2, 3]),
        tensor_positions=True,
        replay_storage=False,
    )
    assert batch.hiddens.shape == (2, 11, 0)
    assert batch.action_mask.sum() >= 2
    wrapper.step_core = original


def test_eval_only_sampling_is_policy_identical_to_replay_rollout():
    wrapper = _wrapper()
    prompt_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    def run(replay_storage: bool):
        torch.manual_seed(23)
        return rollout_continuations(
            wrapper,
            prompt_ids,
            max_new_tokens=4,
            max_stream_steps=16,
            temperature=1.0,
            top_p=0.7,
            generator=torch.Generator().manual_seed(29),
            replay_storage=replay_storage,
        )

    replay = run(True)
    evaluation = run(False)
    for field in (
        "kind",
        "token_ids",
        "action_mask",
    ):
        assert torch.equal(getattr(evaluation, field), getattr(replay, field))
    assert evaluation.hiddens.shape[-1] == 0
    assert replay.hiddens.shape[-1] == KWARGS["model_dim"]


def test_evaluation_compile_failure_restarts_eager_and_restores_capture(monkeypatch):
    wrapper = _wrapper()
    original = wrapper.step_core

    class _Tokenizer:
        def eos_id(self) -> int:
            return 5

        def bos_id(self) -> int:
            return -1

        def encode(self, text: str) -> list[int]:
            return [1, 2, 3]

        def decode(self, ids: list[int]) -> str:
            return "Answer: 42"

    def compiled_proxy(*args, **kwargs):
        return original(*args, **kwargs)

    rows = [
        {
            "prompt": [{"content": f"q-{index}"}],
            "reward_model": {"ground_truth": "42"},
        }
        for index in range(4)
    ]
    import postraining.latent_eval as evaluator

    rollout = evaluator.rollout_continuations
    compiled_calls = 0
    compaction_modes: list[tuple[bool, bool]] = []

    def fail_after_partial_capture(wrapper_arg, *args, **kwargs):
        nonlocal compiled_calls
        compaction_modes.append(
            (
                kwargs["compact_finished"],
                kwargs["finished_batch_size"] == 16,
            )
        )
        if wrapper_arg.step_core is compiled_proxy:
            compiled_calls += 1
            if compiled_calls == 2:
                raise torch._dynamo.exc.Unsupported("test-only compile failure")
        return rollout(wrapper_arg, *args, **kwargs)

    monkeypatch.setattr(
        evaluator, "rollout_continuations", fail_after_partial_capture
    )
    attempts: list[dict[str, object]] = []
    with pytest.warns(RuntimeWarning, match="rerunning this evaluation eager"):
        metrics = evaluate_aime_latent(
            wrapper, _Tokenizer(), rows, samples=4, max_new_tokens=2,
            max_stream_steps=8, chunk=4, seed=5, device=torch.device("cpu"),
            prompt_tokens=8, batch_trajectories=4,
            compiled_step_core=compiled_proxy, captured_attempts=attempts,
        )
    assert metrics["samples"] == 16
    assert len(attempts) == 16
    assert [
        (attempt["problem_index"], attempt["sample_index"])
        for attempt in attempts
    ] == [(problem, sample) for problem in range(4) for sample in range(4)]
    assert wrapper.step_core == original
    assert compaction_modes[:2] == [(True, True), (True, True)]
    assert compaction_modes[-1] == (False, False)
    assert metrics["compiled"] is False
    assert metrics["compile_fallback"] is True
    assert metrics["finished_compaction"] == "none"
    calls_after_failure = compiled_calls

    second_metrics = evaluate_aime_latent(
        wrapper, _Tokenizer(), rows, samples=4, max_new_tokens=2,
        max_stream_steps=8, chunk=4, seed=5, device=torch.device("cpu"),
        prompt_tokens=8, batch_trajectories=4,
        compiled_step_core=compiled_proxy,
    )
    assert compiled_calls == calls_after_failure
    assert second_metrics["compiled"] is False
    assert second_metrics["compile_fallback"] is False


def test_value_only_update_touches_only_the_critic():
    wrapper = _wrapper()
    critic = _critic()
    backbone = wrapper.backbone
    batch = _rollout(wrapper, batch=2)
    assign_terminal_rewards(batch, torch.rand(2))
    critic_before = [p.clone() for p in critic.parameters()]
    policy_before = [p.clone() for p in backbone.policy_probe.parameters()]
    optimizers = {
        "critic": torch.optim.AdamW(critic.parameters(), lr=1e-2, weight_decay=0.0),
    }
    update_minibatch(wrapper, critic, batch, optimizers, value_only=True)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(critic_before, critic.parameters())
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(policy_before, backbone.policy_probe.parameters())
    )


def test_rollout_stops_rows_at_any_stop_token_and_records_nothing_after():
    wrapper = _wrapper()
    stop_ids = (5, 1)
    prompt_ids = torch.randint(0, 32, (8, 4))
    generator = torch.Generator().manual_seed(13)
    batch = trim_stream(
        rollout_continuations(
            wrapper, prompt_ids, 16, 64, 1.0, 1.0,
            generator=generator, stop_ids=stop_ids,
        )
    )
    rows = emitted_token_rows(batch)
    # Uniform-ish sampling over 32 pieces across 8 rows x up to 16 emits
    # makes a stop-token hit near-certain; guard so the assertions bite.
    hits = [
        next((token for token in row if token in stop_ids), None) for row in rows
    ]
    assert any(hit is not None for hit in hits)
    for index, (row, hit) in enumerate(zip(rows, hits)):
        if hit is None:
            continue
        # The stop emit is the final non-pad slot: the row went inactive, so
        # no thinks, emits, or gate decisions follow it.
        assert row[-1] == hit
        alive = batch.kind[index] != PAD_SLOT
        last = int(alive.nonzero().max())
        assert batch.kind[index, last] == TOKEN_SLOT
        assert int(batch.token_ids[index, last]) == hit
        assert not batch.action_mask[index, last + 1:].any()


def test_math_prompt_sampler_is_sequential_and_resumable_in_epoch_zero():
    rows = [{"id": index} for index in range(7)]
    full = MathPromptSampler(rows, seed=3).next_rows(7)
    assert [row["id"] for row in full] == list(range(7))
    resumed = MathPromptSampler(rows, seed=3)
    resumed.cursor = 4
    assert resumed.next_rows(3) == full[4:]
    # The first epoch's order has no RNG dependence.
    assert MathPromptSampler(rows, seed=4).next_rows(7) == full


def test_math_prompt_sampler_wraps_epochs_with_deterministic_reshuffles():
    rows = [{"id": index} for index in range(7)]
    sampler = MathPromptSampler(rows, seed=3)
    # A single request spanning three epochs: each epoch is a permutation of
    # the full dataset, epoch 0 is the given order, and later epochs differ.
    stream = [row["id"] for row in sampler.next_rows(21)]
    epochs = [stream[0:7], stream[7:14], stream[14:21]]
    assert epochs[0] == list(range(7))
    for epoch in epochs[1:]:
        assert sorted(epoch) == list(range(7))
    assert epochs[1] != epochs[0]
    assert sampler.cursor == 21
    assert sampler.epoch == 3
    # Epoch order depends only on (seed, epoch): a resume mid-epoch continues
    # the identical stream, and a different seed reshuffles differently.
    resumed = MathPromptSampler(rows, seed=3)
    resumed.cursor = 10
    assert [row["id"] for row in resumed.next_rows(11)] == stream[10:]
    other_seed = [row["id"] for row in MathPromptSampler(rows, seed=4).next_rows(21)]
    assert other_seed[7:] != stream[7:]


def test_math_dataset_identity_binds_bytes_exclusions_and_order(tmp_path):
    dataset = tmp_path / "math.parquet"
    dataset.write_bytes(b"first")
    original = math_dataset_identity(dataset, "")
    assert math_dataset_identity(dataset, "") == original
    assert math_dataset_identity(dataset, "algebra") != original
    dataset.write_bytes(b"second")
    assert math_dataset_identity(dataset, "") != original


def test_checkpoint_records_partial_value_warmup_for_exact_resume(tmp_path):
    from postraining.math_prompt import ANSWER_FENCE_PROMPT_SCHEMA

    wrapper = _wrapper()
    critic = _critic()
    optimizers = _optimizers(wrapper, critic)
    sampler = MathPromptSampler([{"id": index} for index in range(4)], seed=3)
    sampler.next_rows(3)
    checkpoint = tmp_path / "warmup.pt"
    save_checkpoint(
        checkpoint,
        wrapper,
        critic,
        optimizers,
        step=0,
        args=SimpleNamespace(value_warmup_steps=50, answer_fence=True),
        sampler=sampler,
        warmup_step=20,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["step"] == 0
    assert payload["value_warmup_step"] == 20
    assert payload["sampler_cursor"] == 3
    assert payload["execution_schema"] == EXECUTION_SCHEMA
    assert payload["replay_numerics_schema"] == REPLAY_NUMERICS_SCHEMA
    assert payload["reward_schema"] == REWARD_SCHEMA
    assert payload["actor_objective_schema"] == ACTOR_OBJECTIVE_SCHEMA
    assert payload["answer_fence_prompt_schema"] == ANSWER_FENCE_PROMPT_SCHEMA
    assert payload["thought_input_schema"] == THOUGHT_INPUT_SCHEMA
    assert "torch_adamw" in payload["optimizer_schema"]


def test_value_support_geometry_matches_compares_args_not_shapes() -> None:
    current = SimpleNamespace(
        value_anchored_support=True,
        value_bins=101,
        value_margin_bins=4,
        value_sigma_ratio=1.0,
    )
    saved = {
        "value_anchored_support": True,
        "value_bins": 101,
        "value_margin_bins": 4,
        "value_sigma_ratio": 1.0,
    }
    assert value_support_geometry_matches(saved, current)
    # Anchored 103/3 collides with 101/4 on total head width (110 bins) but
    # changes bin width and sigma; the args comparison catches what a strict
    # state-dict shape check cannot.
    assert not value_support_geometry_matches(
        dict(saved, value_bins=103, value_margin_bins=3), current
    )
    assert not value_support_geometry_matches(
        dict(saved, value_anchored_support=False), current
    )
    assert not value_support_geometry_matches(
        dict(saved, value_sigma_ratio=2.0), current
    )
    # Pre-v22 checkpoints carry none of the keys.
    assert not value_support_geometry_matches({}, current)
    legacy = SimpleNamespace(
        value_anchored_support=False,
        value_bins=101,
        value_margin_bins=4,
        value_sigma_ratio=1.0,
    )
    # Unanchored grids ignore the margin flag entirely.
    assert value_support_geometry_matches(
        {
            "value_anchored_support": False,
            "value_bins": 101,
            "value_margin_bins": 9,
            "value_sigma_ratio": 1.0,
        },
        legacy,
    )
    assert not value_support_geometry_matches({}, legacy)


def test_score_math_rollout_requires_termination_before_verifier_reward(monkeypatch):
    wrapper = _wrapper()
    eos = 5
    batch = _rollout(wrapper, batch=4, prompt=4, new_tokens=8)

    class _Tokenizer:
        def __init__(self):
            self.calls = []

        def decode(self, ids: list[int]) -> str:
            self.calls.append(list(ids))
            return "Answer: 42"

    tokenizer = _Tokenizer()
    import postraining.train_latent_vapo as trainer

    monkeypatch.setattr(
        trainer,
        "emitted_token_rows",
        lambda _: [[1, eos, 9], [1, 2, 3], [eos], [4, eos]],
    )
    score_math_rollout(batch, "42", tokenizer, (eos,))
    assert batch.reward_scalar.tolist() == [1.0, 0.0, 1.0, 1.0]
    # Unterminated rows are rejected without calling the answer verifier;
    # terminated rows are decoded only through the first stop token.
    assert len(tokenizer.calls) == 3
    for ids in tokenizer.calls:
        assert eos not in ids[:-1]
    # Rewards land once per row, on the final gate-decision position.
    assert torch.equal(batch.rewards.sum(dim=1), batch.reward_scalar)
    positions = batch.action_mask.size(1) - 1 - batch.action_mask.flip(1).argmax(1)
    assert torch.equal(
        batch.rewards.gather(1, positions.unsqueeze(1)).squeeze(1),
        batch.reward_scalar,
    )
    score_math_rollout(batch, "7", tokenizer, (eos,))
    partial = nearby_numeric_reward("42", "7")
    assert batch.reward_scalar.tolist() == pytest.approx(
        [partial, 0.0, partial, partial]
    )


def test_score_math_rollout_uses_only_the_final_answer_for_nearby_reward(
    monkeypatch,
):
    wrapper = _wrapper()
    eos = 5
    batch = _rollout(wrapper, batch=3, prompt=4, new_tokens=4)

    class _Tokenizer:
        def decode(self, ids: list[int]) -> str:
            return {
                1: "Answer: 999\nAnswer: 5.5",
                2: "Answer: 999\nAnswer: prose",
                3: "Answer: 5.5\nAnswer: 999",
            }[ids[0]]

    import postraining.train_latent_vapo as trainer

    monkeypatch.setattr(
        trainer,
        "emitted_token_rows",
        lambda _: [[1, eos], [2, eos], [3, eos]],
    )
    score_math_rollout(batch, "5", _Tokenizer(), (eos,))
    assert batch.reward_scalar.tolist() == pytest.approx(
        [nearby_numeric_reward("5.5", "5"), 0.0, nearby_numeric_reward("999", "5")]
    )


class _PositionalTokenizer:
    """Positionally faithful decode: fences and unknowns vanish, each
    known token contributes a fixed piece, so text offsets line up with
    token order (what the positional gate check measures)."""

    PIECES = {
        1: "hmm",
        9: "Answer: 42\n",
        10: "answer : 42\n",  # case/space variant the verifier accepts
        11: "Answer:",  # instruction echo, no graded value after it
        12: "42",  # bare value (the fenced-answer form)
        13: "37",  # bare wrong value
    }

    def decode(self, ids: list[int]) -> str:
        return "".join(self.PIECES.get(token, "") for token in ids)


def test_score_math_rollout_think_fence_gates_all_reward(monkeypatch):
    from postraining.train_latent_vapo import think_format_ok

    wrapper = _wrapper()
    eos, think_open, think_close = 5, 7, 8
    batch = _rollout(wrapper, batch=7, prompt=4, new_tokens=10)
    _Tokenizer = _PositionalTokenizer

    import postraining.train_latent_vapo as trainer

    monkeypatch.setattr(
        trainer,
        "emitted_token_rows",
        lambda _: [
            [think_open, 1, think_close, 9, eos],  # well-formed
            [9, eos],  # bare guess: correct answer, no fence
            [think_open, 1, 9, eos],  # never closed
            [think_open, 1, eos, think_close, 9],  # closed only after stop
            [think_open, think_open, 1, think_close, 9, eos],  # double open
            [think_open, think_close, 9, eos],  # EMPTY fence ritual
            [9, think_open, 1, think_close, eos],  # guess first, fence later
        ],
    )
    score_math_rollout(
        batch,
        "42",
        _Tokenizer(),
        (eos,),
        nearby_reward_max=0.0,
        think_fence_ids=(think_open, think_close),
    )
    assert batch.reward_scalar.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # Gate-zeroed AND verifier-correct: all failing rows except the one
    # whose visible slice never contains the answer token (closed after
    # stop cuts the 9 away).
    assert batch.think_gate_zeroed_correct == 5
    # Without the fence gate the same rows keep their verifier reward.
    score_math_rollout(batch, "42", _Tokenizer(), (eos,))
    assert batch.reward_scalar.tolist() == [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    fence = (think_open, think_close)
    tokenizer = _Tokenizer()
    # The empty fence and the answer-before-close ritual both fail.
    assert not think_format_ok([think_open, think_close], fence, tokenizer)
    assert not think_format_ok([9, think_open, 1, think_close], fence, tokenizer)
    assert not think_format_ok([think_close, 1, think_open], fence, tokenizer)
    assert think_format_ok([think_open, 1, think_close, 9], fence, tokenizer)
    # Red-team HIGH: the gate must match the verifier's own pattern —
    # a case/whitespace variant answer before the fence is still the
    # graded answer and must fail.
    assert not think_format_ok(
        [10, think_open, 1, think_close], fence, tokenizer
    )
    # An Answer: echo INSIDE the fence is not the graded (last) match;
    # zeroing correct rollouts over it would make the gate the problem.
    assert think_format_ok(
        [think_open, 11, think_close, 9], fence, tokenizer
    )
    # Variant answer before the fence loses to a graded answer after
    # the close: the LAST match is what the verifier grades.
    assert think_format_ok(
        [10, think_open, 1, think_close, 9], fence, tokenizer
    )


def test_think_min_tokens_floor_gates_short_fences(monkeypatch):
    from postraining.train_latent_vapo import think_format_ok, think_span_tokens

    wrapper = _wrapper()
    eos, think_open, think_close = 5, 7, 8
    fence = (think_open, think_close)
    tokenizer = _PositionalTokenizer()
    three_inner = [think_open, 1, 2, 3, think_close, 9]
    assert think_span_tokens(three_inner, fence) == 3
    assert think_span_tokens([1, 9], fence) is None
    assert think_span_tokens([think_close, 1, think_open], fence) is None
    assert think_format_ok(three_inner, fence, tokenizer, min_think_tokens=3)
    assert not think_format_ok(three_inner, fence, tokenizer, min_think_tokens=4)
    # min_think_tokens below 1 never readmits the empty-fence ritual.
    assert not think_format_ok(
        [think_open, think_close, 9], fence, tokenizer, min_think_tokens=0
    )

    batch = _rollout(wrapper, batch=2, prompt=4, new_tokens=10)
    import postraining.train_latent_vapo as trainer

    monkeypatch.setattr(
        trainer,
        "emitted_token_rows",
        lambda _: [
            [think_open, 1, 2, 3, think_close, 9, eos],
            [think_open, 1, think_close, 9, eos],  # under the floor
        ],
    )
    score_math_rollout(
        batch,
        "42",
        tokenizer,
        (eos,),
        nearby_reward_max=0.0,
        think_fence_ids=fence,
        min_think_tokens=3,
    )
    assert batch.reward_scalar.tolist() == [1.0, 0.0]
    assert batch.think_gate_zeroed_correct == 1


def test_structural_answer_fence_gate(monkeypatch):
    from postraining.train_latent_vapo import structural_format_ok

    wrapper = _wrapper()
    eos, think_open, think_close = 5, 7, 8
    answer_open, answer_close = 15, 16
    think = (think_open, think_close)
    answer = (answer_open, answer_close)
    tokenizer = _PositionalTokenizer()

    # The gate's contract: tokens are the stop-terminated visible slice,
    # <think> opens the completion, </answer> sits just before the stop.
    honest = [think_open, 1, think_close, answer_open, 12, answer_close, eos]
    assert structural_format_ok(honest, think, answer)
    # Text between </think> and <answer> is admissible (comes after the
    # paid think budget, so it cannot pre-commit around the floor).
    assert structural_format_ok(
        [think_open, 1, think_close, 1, answer_open, 12, answer_close, eos],
        think, answer,
    )
    # Every structural violation fails on token ids alone.
    assert not structural_format_ok(
        [think_open, 1, think_close, 12, eos], think, answer
    )  # no answer fence
    assert not structural_format_ok(
        [think_open, 1, think_close, answer_open, answer_close, eos],
        think, answer,
    )  # empty answer span
    assert not structural_format_ok(
        [answer_open, 12, answer_close, think_open, 1, think_close, eos],
        think, answer,
    )  # guess first, think later
    assert not structural_format_ok(
        [9, think_open, 1, think_close, answer_open, 12, answer_close, eos],
        think, answer,
    )  # red-team: plain-text guess BEFORE <think> — the collapse layout
    assert not structural_format_ok(
        [think_open, 1, think_close, answer_open, 12, answer_close, 13, eos],
        think, answer,
    )  # red-team: trailing junk after </answer> is not free reward
    assert not structural_format_ok(
        [think_open, answer_open, 12, answer_close, 1, think_close, eos],
        think, answer,
    )  # answer inside the think span
    assert not structural_format_ok(
        [think_open, 1, think_close, answer_open, 12, answer_close,
         answer_open, 13, answer_close, eos],
        think, answer,
    )  # duplicated answer fences: no unambiguous graded span
    assert not structural_format_ok(
        honest, think, answer, min_think_tokens=2
    )  # think floor still applies

    import postraining.train_latent_vapo as trainer

    batch = _rollout(wrapper, batch=6, prompt=4, new_tokens=12)
    monkeypatch.setattr(
        trainer,
        "emitted_token_rows",
        lambda _: [
            honest,
            # Decoded-text "Answer: 42" committed before <think> with a
            # WRONG fenced value: anchor kills it, and the relaxed
            # counterfactual grades the FENCED 37 (the policy's answer
            # under fence semantics), so the alarm does not count it.
            [9, think_open, 1, think_close, answer_open, 13, answer_close,
             eos],
            # Correct plain-text answer, no answer fence at all: the
            # relaxed scan finds nothing, the plain parse counts it.
            [think_open, 1, think_close, 9, eos],
            # Fence-native collapse: correct value inside an answer span
            # with NO think fence. decode strips the fences, so only the
            # relaxed counterfactual can see the value (red-team: this
            # read as a clean zero and hid the collapse).
            [answer_open, 12, answer_close, eos],
            # Correct fenced value, junk after the close: anchored gate
            # zeroes it; relaxed counterfactual counts it.
            [think_open, 1, think_close, answer_open, 12, answer_close, 13,
             eos],
            # Unterminated row scores zero before any gate runs.
            honest[:-1],
        ],
    )
    score_math_rollout(
        batch,
        "42",
        tokenizer,
        (eos,),
        nearby_reward_max=0.0,
        think_fence_ids=think,
        min_think_tokens=1,
        answer_fence_ids=answer,
    )
    assert batch.reward_scalar.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert batch.think_gate_zeroed_correct == 3

    with pytest.raises(ValueError, match="requires think_fence_ids"):
        score_math_rollout(
            batch, "42", tokenizer, (eos,), answer_fence_ids=answer
        )


def test_fenced_answer_window_and_relaxed_extraction(monkeypatch):
    from postraining.train_latent_vapo import relaxed_fenced_answer_text

    wrapper = _wrapper()
    eos, think_open, think_close = 5, 7, 8
    answer_open, answer_close = 15, 16
    think = (think_open, think_close)
    answer = (answer_open, answer_close)

    class _LongTokenizer:
        # 20: 299 chars of leading whitespace then the correct value —
        # under the verifier's default 300-char tail window the
        # synthesized "Answer: " prefix falls outside the window and the
        # row grades [INVALID]; the field pattern's \s* consumes the
        # padding, so with window=None the value verifies.
        PIECES = {1: "hmm", 20: " " * 297 + "42"}

        def decode(self, ids: list[int]) -> str:
            return "".join(self.PIECES.get(token, "") for token in ids)

    tokenizer = _LongTokenizer()
    import postraining.train_latent_vapo as trainer

    batch = _rollout(wrapper, batch=1, prompt=4, new_tokens=12)
    monkeypatch.setattr(
        trainer,
        "emitted_token_rows",
        lambda _: [
            [think_open, 1, think_close, answer_open, 20, answer_close, eos],
        ],
    )
    score_math_rollout(
        batch, "42", tokenizer, (eos,),
        nearby_reward_max=0.0,
        think_fence_ids=think,
        answer_fence_ids=answer,
    )
    # Red-team: a structurally valid long answer must not fall off the
    # verifier's tail window into a silent zero.
    assert batch.reward_scalar.tolist() == [1.0]

    # Relaxed extraction tolerates duplication (grades the FIRST span),
    # missing think fences, and unanchored layouts — and only breakage
    # with no recoverable span returns None.
    long_piece = _LongTokenizer()
    assert relaxed_fenced_answer_text(
        [answer_open, 20, answer_close, answer_open, 1, answer_close],
        long_piece, answer,
    ) == " " * 297 + "42"
    assert relaxed_fenced_answer_text([answer_open, 20], long_piece, answer) is None
    assert relaxed_fenced_answer_text([20], long_piece, answer) is None
    assert relaxed_fenced_answer_text(
        [answer_open, answer_close], long_piece, answer
    ) is None
    # Documented over-count: a scratch <answer> span INSIDE the think
    # span wins over the policy's actual (post-think) answer, so the
    # gate-zeroed-correct alarm reads approximate, not exact.
    assert relaxed_fenced_answer_text(
        [
            think_open, answer_open, 20, answer_close, think_close,
            answer_open, 1, answer_close, eos,
        ],
        long_piece, answer,
    ) == " " * 297 + "42"


def test_rollout_diagnostics_reports_think_format_fraction(monkeypatch):
    from postraining.train_latent_vapo import rollout_diagnostics

    wrapper = _wrapper()
    eos, think_open, think_close = 5, 7, 8
    batch = _rollout(wrapper, batch=4, prompt=4, new_tokens=8)
    import postraining.train_latent_vapo as trainer

    class _Tokenizer:
        def decode(self, ids: list[int]) -> str:
            return "thinking text"

    monkeypatch.setattr(
        trainer,
        "emitted_token_rows",
        lambda _: [
            [think_open, 1, think_close, eos],
            [1, 2, eos],
            # Unterminated rows earn no reward, so a well-formed fence
            # there is NOT compliant (its length still counts below).
            [think_open, 1, think_close, 2],
            # The fence completed after the stop cut is invisible.
            [think_open, eos, think_close],
        ],
    )
    metrics = rollout_diagnostics(
        batch,
        samples_per_prompt=2,
        stop_ids=(eos,),
        refreshed_statistics=False,
        think_fence_ids=(think_open, think_close),
        tokenizer=_Tokenizer(),
    )
    assert metrics["think_format_fraction"] == pytest.approx(0.25)
    # Inner lengths over structurally intact fences on the visible slice:
    # rows 1 and 3 have one token inside; row 2 has no fence and the
    # stop-cut row keeps only an unclosed open. Mean = 1.0.
    assert metrics["think_tokens_mean"] == pytest.approx(1.0)
    assert metrics["think_tokens_sum"] == pytest.approx(2.0)
    assert metrics["think_tokens_count"] == pytest.approx(2.0)
    # No scorer stamped this batch: absence, not a healthy-looking 0.0.
    assert "think_gate_zeroed_correct_fraction" not in metrics
    floored = rollout_diagnostics(
        batch,
        samples_per_prompt=2,
        stop_ids=(eos,),
        refreshed_statistics=False,
        think_fence_ids=(think_open, think_close),
        tokenizer=_Tokenizer(),
        min_think_tokens=2,
    )
    # The same rows fall below a 2-token floor.
    assert floored["think_format_fraction"] == 0.0
    assert floored["think_tokens_mean"] == pytest.approx(1.0)
    without = rollout_diagnostics(
        batch, samples_per_prompt=2, stop_ids=(eos,), refreshed_statistics=False
    )
    assert "think_format_fraction" not in without
    with pytest.raises(ValueError, match="requires the tokenizer"):
        rollout_diagnostics(
            batch,
            samples_per_prompt=2,
            stop_ids=(eos,),
            refreshed_statistics=False,
            think_fence_ids=(think_open, think_close),
        )


def test_aggregate_diagnostics_tolerates_heterogeneous_gate_state(monkeypatch):
    from postraining.train_latent_vapo import aggregate_diagnostics

    wrapper = _wrapper()
    eos, think_open, think_close = 5, 7, 8
    scored = _rollout(wrapper, batch=2, prompt=4, new_tokens=8)
    unscored = _rollout(wrapper, batch=2, prompt=4, new_tokens=8)
    # One group carries a scorer-stamped gate count, the other does not
    # (its key is conditionally absent). Aggregation must key on the
    # intersection: per_group[0]-driven indexing would KeyError in this
    # order and silently drop metrics in the reverse.
    scored.think_gate_zeroed_correct = 1
    unscored.think_gate_zeroed_correct = None
    import postraining.train_latent_vapo as trainer

    class _Tokenizer:
        def decode(self, ids: list[int]) -> str:
            return "thinking text"

    monkeypatch.setattr(
        trainer,
        "emitted_token_rows",
        lambda _: [[think_open, 1, think_close, eos], [1, eos]],
    )
    for order in ((scored, unscored), (unscored, scored)):
        metrics = aggregate_diagnostics(
            list(order),
            samples_per_prompt=2,
            stop_ids=(eos,),
            refreshed_statistics=False,
            think_fence_ids=(think_open, think_close),
            tokenizer=_Tokenizer(),
        )
        assert metrics["trajectories"] == 4
        assert "think_gate_zeroed_correct_fraction" not in metrics
        assert metrics["think_format_fraction"] == pytest.approx(0.5)


def test_load_posttraining_tokenizer_think_tokens():
    from postraining.core import load_posttraining_tokenizer

    tokenizer = load_posttraining_tokenizer(
        "nanogpt_mini_gpt2vocab_kda_kkkdkkkd_mixers_v3", "", think_tokens=True
    )
    assert tokenizer.think_open_id == 50257
    assert tokenizer.think_close_id == 50258
    assert tokenizer.answer_open_id is None
    assert tokenizer.answer_close_id is None
    with pytest.raises(ValueError, match="padded-vocab slack"):
        load_posttraining_tokenizer("fresh_lejepa", "", think_tokens=True)


def test_load_posttraining_tokenizer_answer_tokens():
    from postraining.core import GPT2BPETokenizer, load_posttraining_tokenizer

    tokenizer = load_posttraining_tokenizer(
        "nanogpt_mini_gpt2vocab_kda_kkkdkkkd_mixers_v3",
        "",
        think_tokens=True,
        answer_tokens=True,
    )
    # Registration order fixes the ids: think pair first, answer pair in
    # the next two padded-vocab slack rows.
    assert tokenizer.think_open_id == 50257
    assert tokenizer.think_close_id == 50258
    assert tokenizer.answer_open_id == 50259
    assert tokenizer.answer_close_id == 50260
    # All four are specials: decode drops them so text parsing never
    # sees fence markup.
    row = tokenizer.encode("x") + [50259] + tokenizer.encode("7") + [50260]
    assert tokenizer.decode(row) == "x7"
    # Fenced extraction returns exactly the decoded inner span; broken or
    # empty structures return None.
    from postraining.core import fenced_answer_text, single_fence_span

    assert single_fence_span(row, (50259, 50260)) == (1, 3)
    assert fenced_answer_text(row, tokenizer, (50259, 50260)) == "7"
    assert fenced_answer_text(row + [50260], tokenizer, (50259, 50260)) is None
    assert fenced_answer_text(
        [50259, 50260], tokenizer, (50259, 50260)
    ) is None
    assert fenced_answer_text([50260, 50259], tokenizer, (50259, 50260)) is None
    with pytest.raises(ValueError, match="requires think_tokens"):
        GPT2BPETokenizer(answer_tokens=True)
    with pytest.raises(ValueError, match="padded-vocab slack"):
        load_posttraining_tokenizer(
            "fresh_lejepa", "", think_tokens=True, answer_tokens=True
        )


def test_verify_terminated_answer_rejects_cap_truncation():
    class _Tokenizer:
        def decode(self, ids: list[int]) -> str:
            return "Answer: 42"

    tokenizer = _Tokenizer()
    assert verify_terminated_answer([1, 2, 3], "42", tokenizer, (5, 4)) == (
        False,
        "[UNTERMINATED]",
    )
    assert verify_terminated_answer([1, 5, 3], "42", tokenizer, (5, 4))[0]
    assert verify_terminated_answer([1, 4, 3], "42", tokenizer, (5, 4))[0]


def test_verify_terminated_answer_grades_fenced_span():
    class _Tokenizer:
        PIECES = {1: "Answer: 37\n", 2: "42"}

        def decode(self, ids: list[int]) -> str:
            return "".join(self.PIECES.get(token, "") for token in ids)

    tokenizer = _Tokenizer()
    stop, answer_open, answer_close = 5, 15, 16
    fence = (answer_open, answer_close)
    # The fenced span is graded even when misleading plain text precedes
    # it; the plain-text "Answer: 37" is structurally outside the fence.
    correct, _ = verify_terminated_answer(
        [1, answer_open, 2, answer_close, stop], "42", tokenizer, (stop,),
        answer_fence_ids=fence,
    )
    assert correct
    # No fence: falls back to the plain-text parse so raw correctness
    # keeps registering (structure enforcement is the RL gate's job).
    wrong, _ = verify_terminated_answer(
        [1, stop], "42", tokenizer, (stop,), answer_fence_ids=fence
    )
    assert not wrong
    plain_correct, _ = verify_terminated_answer(
        [2, stop], "37", tokenizer, (stop,), answer_fence_ids=fence
    )
    # "42" alone has no Answer: field — the fallback grades the decoded
    # text with the standard parser, exactly as a fenceless eval would.
    assert not plain_correct


def test_evaluate_aime_latent_threads_stop_ids_into_the_rollout(monkeypatch):
    wrapper = _wrapper()

    class _Tokenizer:
        def __init__(self, eos: int, bos: int):
            self.eos = eos
            self.bos = bos

        def eos_id(self) -> int:
            return self.eos

        def bos_id(self) -> int:
            return self.bos

        def encode(self, text: str) -> list[int]:
            return [1, 2, 3]

        def decode(self, ids: list[int]) -> str:
            return "Answer: 42"

    rows = [{"prompt": [{"content": "q"}], "reward_model": {"ground_truth": "42"}}]
    seen = []

    def _spy(*args, **kwargs):
        seen.append(kwargs.get("stop_ids"))
        return rollout_continuations(*args, **kwargs)

    import postraining.latent_eval as evaluator

    monkeypatch.setattr(evaluator, "rollout_continuations", _spy)
    # Both boundary tokens must reach the rollout so eval rows stop exactly
    # like training rollouts; -1 sentinels drop out.
    for eos, bos, expected in ((5, 4, (5, 4)), (5, -1, (5,))):
        evaluate_aime_latent(
            wrapper, _Tokenizer(eos, bos), rows, samples=2, max_new_tokens=2,
            max_stream_steps=8, chunk=2, seed=5, device=torch.device("cpu"),
            prompt_tokens=8,
        )
        assert seen and seen[-1] == expected
    with pytest.raises(RuntimeError, match="requires a valid BOS or EOS"):
        evaluate_aime_latent(
            wrapper, _Tokenizer(-1, -1), rows, samples=2, max_new_tokens=2,
            max_stream_steps=8, chunk=2, seed=5, device=torch.device("cpu"),
            prompt_tokens=8,
        )


def test_evaluate_aime_latent_keeps_the_prompt_tail(monkeypatch):
    wrapper = _wrapper()

    class _Tokenizer:
        def eos_id(self) -> int:
            return -1

        def bos_id(self) -> int:
            return 21

        def encode(self, text: str) -> list[int]:
            return list(range(20))

        def decode(self, ids: list[int]) -> str:
            return "Answer: 42"

    rows = [{"prompt": [{"content": "q"}], "reward_model": {"ground_truth": "42"}}]
    seen: list[torch.Tensor] = []

    def _spy(wrapper_arg, prompt_ids, *args, **kwargs):
        seen.append(prompt_ids.clone())
        return rollout_continuations(wrapper_arg, prompt_ids, *args, **kwargs)

    import postraining.latent_eval as evaluator

    monkeypatch.setattr(evaluator, "rollout_continuations", _spy)
    # The question and answer-format instruction sit at the END of DAPO/AIME
    # prompts, so truncation must keep the tail — behind the BOS that frames
    # the prompt as a document start, matching the training path.
    evaluate_aime_latent(
        wrapper, _Tokenizer(), rows, samples=2, max_new_tokens=2,
        max_stream_steps=8, chunk=2, seed=5, device=torch.device("cpu"),
        prompt_tokens=5,
    )
    assert seen and seen[-1].shape == (1, 5)
    assert seen[-1][0].tolist() == [21] + list(range(16, 20))


def test_key_mask_step_matches_the_narrow_step_over_a_padded_cache():
    wrapper = _wrapper()
    device = torch.device("cpu")
    batch, prompt, padding = 2, 6, 5
    torch.manual_seed(21)
    prompt_ids = torch.randint(0, 32, (batch, prompt))
    tight = wrapper.make_generation_cache(batch, prompt, device)
    padded = wrapper.make_static_generation_cache(batch, prompt + padding, device)
    position_index = torch.zeros((), dtype=torch.long)
    key_masks = torch.ones(
        (prompt + padding, prompt + padding), dtype=torch.bool
    ).tril_()
    for position in range(prompt):
        reference = wrapper.token_step(prompt_ids[:, position], tight, position)
        position_index.fill_(position)
        masked = wrapper.token_step(
            prompt_ids[:, position], padded, position_index, key_masks[position]
        )
        torch.testing.assert_close(masked.belief, reference.belief)
        torch.testing.assert_close(masked.logits, reference.logits)


def test_generation_cache_dtype_does_not_cast_base_model_master_weights():
    model = FreshLeJEPAGPT(**KWARGS).eval()
    assert {parameter.dtype for parameter in model.parameters()} == {
        torch.float32
    }

    caches = model.make_generation_cache(
        batch_size=2,
        max_length=7,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert all(
        tensor.dtype == torch.bfloat16 for cache in caches for tensor in cache
    )
    assert {parameter.dtype for parameter in model.parameters()} == {
        torch.float32
    }


def test_generation_cache_dtype_flows_through_pope_wrapper_and_static_cache():
    wrapper = _wrapper()
    assert {parameter.dtype for parameter in wrapper.parameters()} == {
        torch.float32
    }

    dynamic = wrapper.make_generation_cache(
        batch_size=2,
        max_length=7,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    static = wrapper.make_static_generation_cache(
        batch_size=2,
        cache_length=7,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    for caches in (dynamic, static):
        assert all(
            tensor.dtype == torch.bfloat16
            for cache in caches
            for tensor in cache
        )
    assert {parameter.dtype for parameter in wrapper.parameters()} == {
        torch.float32
    }


def test_key_mask_stepping_requires_a_tensor_position():
    wrapper = _wrapper()
    caches = wrapper.make_static_generation_cache(1, 4, torch.device("cpu"))
    with pytest.raises(ValueError, match="tensor position"):
        wrapper.token_step(
            torch.zeros(1, dtype=torch.long), caches, 0,
            torch.ones(4, dtype=torch.bool),
        )


def test_static_cache_rollout_matches_the_dynamic_rollout_and_is_reusable():
    wrapper = _wrapper()
    device = torch.device("cpu")
    batch, prompt, new_tokens, stream_steps = 3, 5, 4, 12
    torch.manual_seed(23)
    prompt_ids = torch.randint(0, 32, (batch, prompt))

    def roll(caches):
        return rollout_continuations(
            wrapper, prompt_ids, new_tokens, stream_steps, 1.0, 1.0,
            generator=torch.Generator().manual_seed(9), caches=caches,
        )

    dynamic = roll(None)
    # Longer than needed on purpose: masked slots must never leak into the
    # attention, and the same cache set must be reusable WITHOUT re-zeroing
    # (stale finite values stay masked).
    caches = wrapper.make_static_generation_cache(
        batch, prompt + stream_steps + 3, device
    )
    for static in (roll(caches), roll(caches)):
        assert static.prompt_length == dynamic.prompt_length
        assert torch.equal(static.kind, dynamic.kind)
        assert torch.equal(static.token_ids, dynamic.token_ids)
        assert torch.equal(static.action_mask, dynamic.action_mask)
        torch.testing.assert_close(static.hiddens, dynamic.hiddens)
        torch.testing.assert_close(
            static.old_token_logprobs, dynamic.old_token_logprobs
        )


@pytest.mark.parametrize("top_p", [1.0, 0.8])
def test_explicit_rollout_generator_is_independent_of_global_rng(top_p):
    wrapper = _wrapper()
    prompt_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

    def roll(global_draws: int):
        torch.manual_seed(101)
        torch.rand(global_draws)
        return rollout_continuations(
            wrapper,
            prompt_ids,
            5,
            16,
            1.0,
            top_p,
            generator=torch.Generator().manual_seed(103),
        )

    first = roll(1)
    second = roll(1000)
    assert torch.equal(first.kind, second.kind)
    assert torch.equal(first.token_ids, second.token_ids)
    torch.testing.assert_close(first.hiddens, second.hiddens)


def test_preallocated_caches_that_do_not_fit_are_rejected():
    wrapper = _wrapper()
    device = torch.device("cpu")
    prompt_ids = torch.randint(0, 32, (2, 5))
    small = wrapper.make_static_generation_cache(2, 8, device)
    with pytest.raises(ValueError, match="do not fit"):
        rollout_continuations(
            wrapper, prompt_ids, 4, 12, 1.0, 1.0, caches=small
        )
    too_few = wrapper.make_static_generation_cache(1, 32, device)
    with pytest.raises(ValueError, match="fewer than"):
        rollout_continuations(
            wrapper, prompt_ids, 4, 12, 1.0, 1.0, caches=too_few
        )
    ragged = wrapper.make_static_generation_cache(5, 32, device)
    with pytest.raises(ValueError, match="whole number"):
        rollout_continuations(
            wrapper,
            prompt_ids,
            4,
            12,
            1.0,
            1.0,
            caches=ragged,
            prompt_repeats=2,
        )


def test_bucketed_trim_pads_to_the_multiple_without_changing_the_update():
    def run(multiple: int) -> dict[str, float]:
        wrapper = _wrapper()
        critic = _critic()
        torch.manual_seed(31)
        prompt_ids = torch.randint(0, 32, (2, 5))
        batch = rollout_continuations(
            wrapper, prompt_ids, 4, 32, 1.0, 1.0,
            generator=torch.Generator().manual_seed(7),
        )
        batch = trim_stream(batch, multiple=multiple)
        # The bucket boundary is strict — content reaching into the last
        # partial bucket pads BEYOND the original stream rather than leaking
        # an arbitrary (compile-triggering) length.
        assert batch.stream_length % multiple == 0
        assign_terminal_rewards(batch, torch.tensor([1.0, 0.0]))
        refresh_old_statistics(wrapper, critic, batch)
        return update_minibatch(
            wrapper, critic, batch, _optimizers(wrapper, critic)
        )

    exact = run(1)
    # multiple=64 exceeds the 5 + 32 stream, forcing the pad-beyond path.
    for multiple in (16, 64):
        bucketed = run(multiple)
        # Padding invariance: PAD inputs are zeroed, attention is causal, and
        # every loss is masked — the bucketed stream must train identically.
        for key, value in exact.items():
            assert bucketed[key] == pytest.approx(value, rel=1e-5, abs=1e-6), key


def test_trim_stream_releases_the_full_capacity_backing_storage():
    wrapper = _wrapper()
    prompt_ids = torch.randint(0, 32, (2, 5))
    full = rollout_continuations(
        wrapper,
        prompt_ids,
        max_new_tokens=1,
        max_stream_steps=128,
        temperature=1.0,
        top_p=1.0,
        generator=torch.Generator().manual_seed(7),
    )
    assert full.stream_length == prompt_ids.size(1) + 128

    trimmed = trim_stream(full)
    assert trimmed.stream_length < full.stream_length
    for name in ("kind", "token_ids", "hiddens", "old_values"):
        original = getattr(full, name)
        compact = getattr(trimmed, name)
        assert compact.untyped_storage().data_ptr() != original.untyped_storage().data_ptr(), name
        assert compact.untyped_storage().nbytes() < original.untyped_storage().nbytes(), name


def test_refresh_runs_grad_enabled_but_stores_detached_statistics():
    wrapper = _wrapper()
    critic = _critic()
    batch = _rollout(wrapper)
    refresh_old_statistics(wrapper, critic, batch)
    for name in (
        "old_values",
        "old_token_logprobs",
    ):
        stored = getattr(batch, name)
        assert not stored.requires_grad, name
        assert stored.grad_fn is None, name


def _deterministic_wrapper() -> LatentThoughtModel:
    """Argmax token draws make trajectories RNG-independent."""
    return _wrapper()


def test_left_padded_batched_rollout_matches_the_sequential_rollouts():
    wrapper = _deterministic_wrapper()
    prompts = [
        torch.randint(1, 32, (7,), generator=torch.Generator().manual_seed(11)),
        torch.randint(1, 32, (4,), generator=torch.Generator().manual_seed(12)),
        torch.randint(1, 32, (6,), generator=torch.Generator().manual_seed(13)),
    ]
    samples = 2
    width = max(prompt.numel() for prompt in prompts)
    prompt_ids = torch.zeros((len(prompts) * samples, width), dtype=torch.long)
    for index, prompt in enumerate(prompts):
        rows = slice(index * samples, (index + 1) * samples)
        prompt_ids[rows, width - prompt.numel():] = prompt
    prompt_lengths = torch.tensor(
        [prompt.numel() for prompt in prompts]
    ).repeat_interleave(samples)

    torch.manual_seed(29)
    batched = rollout_continuations(
        wrapper, prompt_ids, 4, 8, 1.0, 1e-6,
        prompt_lengths=prompt_lengths,
    )
    groups = split_rollout_groups(batched, samples, prompt_lengths)

    for group, prompt in zip(groups, prompts, strict=True):
        torch.manual_seed(29)
        expected = trim_stream(
            rollout_continuations(
                wrapper, prompt[None].expand(samples, -1), 4, 8, 1.0, 1e-6,
            )
        )
        # The padded rows shift every position by a constant, and PoPE
        # scores depend only on position differences — after undoing the
        # padding the trajectories must be identical.
        assert group.prompt_length == expected.prompt_length
        assert group.stream_length == expected.stream_length
        assert torch.equal(group.kind, expected.kind)
        assert torch.equal(group.token_ids, expected.token_ids)
        assert torch.equal(group.action_mask, expected.action_mask)
        torch.testing.assert_close(group.hiddens, expected.hiddens)
        torch.testing.assert_close(
            group.old_token_logprobs,
            expected.old_token_logprobs,
            rtol=1e-4,
            atol=1e-5,
        )


def test_zero_padding_batched_rollout_is_identical_to_the_plain_path():
    wrapper = _deterministic_wrapper()
    prompt_ids = torch.randint(
        1, 32, (4, 6), generator=torch.Generator().manual_seed(17)
    )
    torch.manual_seed(31)
    plain = rollout_continuations(wrapper, prompt_ids, 4, 8, 1.0, 1e-6)
    torch.manual_seed(31)
    padded = rollout_continuations(
        wrapper, prompt_ids, 4, 8, 1.0, 1e-6,
        prompt_lengths=torch.full((4,), 6, dtype=torch.long),
    )
    assert torch.equal(plain.kind, padded.kind)
    assert torch.equal(plain.token_ids, padded.token_ids)
    torch.testing.assert_close(
        plain.old_token_logprobs, padded.old_token_logprobs, rtol=1e-4, atol=1e-5
    )


def test_combined_replay_batch_right_pads_without_changing_beliefs():
    wrapper = _deterministic_wrapper()
    samples = 2
    groups = []
    for prompt_length, seed in ((4, 41), (7, 43), (5, 47)):
        prompt = torch.randint(
            1,
            32,
            (samples, prompt_length),
            generator=torch.Generator().manual_seed(seed),
        )
        groups.append(
            trim_stream(
                rollout_continuations(wrapper, prompt, 4, 8, 1.0, 1e-6)
            )
        )
    for source_id, group in enumerate(groups):
        group.source_id = torch.full(
            (samples,), source_id, dtype=torch.long
        )
        group.verifier_status = torch.full(
            (samples,), source_id + 1, dtype=torch.long
        )

    combined = pack_rollout_groups_for_replay(groups)
    assert combined.source_id is not None
    assert combined.source_id.tolist() == [0, 0, 1, 1, 2, 2]
    assert combined.verifier_status is not None
    assert combined.verifier_status.tolist() == [1, 1, 2, 2, 3, 3]
    assert combined.prompt_length == 4
    assert combined.stream_length == max(group.stream_length for group in groups)
    row_start = 0
    for group in groups:
        row_end = row_start + samples
        for name in (
            "kind",
            "token_ids",
            "hiddens",
            "action_mask",
            "rewards",
        ):
            source = getattr(group, name)
            actual = getattr(combined, name)[
                row_start:row_end, : group.stream_length
            ]
            assert torch.equal(actual, source), name
        assert bool(
            (combined.kind[row_start:row_end, group.stream_length:] == PAD_SLOT).all()
        )
        row_start = row_end

    critic = _critic()
    refresh_old_statistics(wrapper, critic, combined)
    scatter_replay_statistics(combined, groups)
    refreshed_repacked = pack_rollout_groups_for_replay(groups)
    metrics = update_minibatch(
        wrapper,
        critic,
        refreshed_repacked,
        _optimizers(wrapper, critic),
        actor_step=False,
        critic_step=False,
    )
    assert metrics["token_abs_log_ratio_max"] == 0.0
    assert metrics["policy_clip_fraction"] == 0.0

    combined.old_token_logprobs.copy_(
        torch.arange(combined.old_token_logprobs.numel()).reshape_as(
            combined.old_token_logprobs
        )
    )
    combined.old_values.copy_(combined.old_token_logprobs + 2)
    scatter_replay_statistics(combined, groups)
    repacked = pack_rollout_groups_for_replay(groups)
    for name in (
        "old_token_logprobs",
        "old_values",
    ):
        expected = getattr(combined, name)
        actual = getattr(repacked, name)
        real = combined.kind != PAD_SLOT
        if expected.dim() == 3:
            real = real[..., None].expand_as(expected)
        assert torch.equal(actual[real], expected[real]), name

    _, combined_beliefs = replay_beliefs(wrapper, combined)
    row_start = 0
    for group in groups:
        row_end = row_start + samples
        _, expected_beliefs = replay_beliefs(wrapper, group)
        torch.testing.assert_close(
            combined_beliefs[row_start:row_end, : group.stream_length],
            expected_beliefs,
            rtol=2e-4,
            atol=2e-5,
        )
        row_start = row_end


def test_pack_rollout_groups_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        pack_rollout_groups_for_replay([])


def test_batched_rollout_rejects_bad_prompt_lengths_and_static_caches():
    wrapper = _wrapper()
    prompt_ids = torch.randint(1, 32, (2, 5))
    with pytest.raises(ValueError, match="one true length per prompt"):
        rollout_continuations(
            wrapper, prompt_ids, 2, 4, 1.0, 1.0,
            prompt_lengths=torch.tensor([5]),
        )
    with pytest.raises(ValueError, match=r"\[1, prompt_ids width\]"):
        rollout_continuations(
            wrapper, prompt_ids, 2, 4, 1.0, 1.0,
            prompt_lengths=torch.tensor([5, 6]),
        )
    caches = wrapper.make_generation_cache(2, 9, torch.device("cpu"))
    # A caller-owned cache with no decode_mask falls back to the shared causal
    # key mask, which has no room for a per-row left pad.
    with pytest.raises(ValueError, match="allocated cache or a decode_mask"):
        rollout_continuations(
            wrapper, prompt_ids, 2, 4, 1.0, 1.0,
            caches=caches, prompt_lengths=torch.tensor([5, 5]),
        )


def test_prompt_repeats_expand_unique_prompt_storage():
    wrapper = _wrapper()
    prompt_ids = torch.tensor([[0, 0, 4, 5, 6], [7, 8, 9, 10, 11]])
    prompt_lengths = torch.tensor([3, 5])
    batch = rollout_continuations(
        wrapper,
        prompt_ids,
        max_new_tokens=0,
        max_stream_steps=1,
        temperature=1.0,
        top_p=1.0,
        prompt_lengths=prompt_lengths,
        prompt_repeats=3,
    )
    assert batch.kind.size(0) == 6
    for prompt_index in range(2):
        members = slice(prompt_index * 3, (prompt_index + 1) * 3)
        expected = prompt_ids[prompt_index].expand(3, -1)
        assert torch.equal(batch.token_ids[members, : prompt_ids.size(1)], expected)


def test_split_rollout_groups_rejects_mixed_lengths_within_a_group():
    wrapper = _deterministic_wrapper()
    prompt_ids = torch.randint(1, 32, (2, 5))
    lengths = torch.tensor([5, 4])
    batch = rollout_continuations(
        wrapper, prompt_ids, 2, 4, 1.0, 1e-6, prompt_lengths=lengths
    )
    with pytest.raises(ValueError, match="share one prompt length"):
        split_rollout_groups(batch, 2, lengths)


def _boolean_mask(rows: int, stream: int, seed: int = 5) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand(rows, stream, generator=generator) < 0.35
    # An all-false and an all-true row: the compaction must survive both.
    mask[0] = False
    mask[-1] = True
    return mask


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_slot_index_compaction_reproduces_boolean_indexing(dtype):
    # The whole point of index_select here is that it gathers the same slots
    # in the same order as tensor[mask], so no numeric result may move.
    mask = _boolean_mask(6, 11)
    index = slot_index(mask)
    generator = torch.Generator().manual_seed(21)
    for shape in ((6, 11), (6, 11, 4)):
        values = torch.randn(*shape, generator=generator).to(dtype)
        assert torch.equal(compact_slots(values, index), values[mask])

    values = torch.randn(6, 11, 4, generator=generator).to(dtype)
    source = compact_slots(values, index)
    by_index = scatter_slots(torch.zeros_like(values), index, source)
    by_mask = torch.zeros_like(values)
    by_mask[mask] = source
    assert torch.equal(by_index, by_mask)
    # masked_scatter fills the same slots from the same flat source, so the
    # replaced call site is exact too.
    assert torch.equal(
        torch.zeros_like(values).masked_scatter(mask[..., None], source),
        by_index,
    )


def test_slot_index_survives_an_empty_mask():
    mask = torch.zeros(3, 4, dtype=torch.bool)
    index = slot_index(mask)
    assert index.numel() == 0
    values = torch.randn(3, 4, 2)
    assert torch.equal(compact_slots(values, index), values[mask])
    assert torch.equal(
        scatter_slots(torch.zeros_like(values), index, values[mask]),
        torch.zeros_like(values),
    )


def test_compact_slots_refuses_a_layout_it_would_have_to_copy():
    # reshape would silently materialize the whole dense tensor here, which
    # is exactly the cost the index path exists to avoid.
    mask = _boolean_mask(4, 6)
    values = torch.randn(6, 4, 3).transpose(0, 1)
    with pytest.raises(RuntimeError):
        compact_slots(values, slot_index(mask))


def test_slot_scatter_carries_gradients_to_the_compact_source():
    mask = _boolean_mask(4, 7)
    index = slot_index(mask)
    source = torch.randn(int(mask.sum()), requires_grad=True)
    destination = scatter_slots(torch.zeros(4, 7), index, source)
    weights = torch.randn(4, 7)
    (destination * weights).sum().backward()
    assert torch.equal(source.grad, weights[mask])


def test_the_shard_plan_matches_the_rows_the_iterator_yields():
    # The planner now uploads every shard's rows in one transfer and hands
    # out views of it; those views must still be the host plan, in order.
    wrapper = _wrapper()
    batch = _rollout(wrapper, batch=6, prompt=5, new_tokens=4)
    plan = plan_length_aware_shards(batch, 2, 4 * 1024 * 1024, 1)
    assert len(plan) > 1
    shards = list(iter_length_aware_microbatches(batch, 2, 4 * 1024 * 1024, 1))
    assert len(shards) == len(plan)
    for (host_rows, length, rows), (microbatch, yielded, yielded_length, yielded_rows) in zip(
        plan, shards, strict=True
    ):
        assert host_rows == yielded_rows
        assert length == yielded_length
        assert torch.equal(rows, torch.tensor(host_rows, dtype=torch.long))
        assert torch.equal(yielded, rows)
        assert microbatch.kind.size(0) == len(host_rows)


def test_replay_plan_matches_canonical_action_masks_and_targets():
    wrapper = _wrapper()
    batch = _rollout(wrapper, batch=7, prompt=5, new_tokens=5)
    plan = build_replay_plan(
        batch,
        max_trajectories=2,
        attention_budget=4 * 1024 * 1024,
        bucket_multiple=4,
    )

    planned_rows = []
    for microbatch, shard in iter_planned_replay_microbatches(batch, plan):
        planned_rows.extend(shard.host_rows)
        torch.testing.assert_close(
            shard.emit_index,
            slot_index(microbatch.action_mask.bool()),
        )
        rows = torch.div(
            shard.emit_index,
            shard.stream_length,
            rounding_mode="floor",
        )
        columns = shard.emit_index.remainder(shard.stream_length)
        assert torch.all(columns + 1 < shard.stream_length)
        assert torch.all(
            microbatch.kind[rows, columns + 1] == TOKEN_SLOT
        )

    assert sorted(planned_rows) == list(range(batch.kind.size(0)))
    assert len(planned_rows) == len(set(planned_rows))


def test_replay_plan_rejects_same_shape_batch_and_different_settings():
    wrapper = _wrapper()
    batch = _rollout(wrapper, batch=4, prompt=5, new_tokens=3)
    plan = build_replay_plan(batch, 2, 4 * 1024 * 1024)
    unrelated = copy.deepcopy(batch)
    unrelated.replay_layout_token = object()

    with pytest.raises(ValueError, match="different packed layout"):
        list(iter_planned_replay_microbatches(unrelated, plan))
    with pytest.raises(ValueError, match="settings"):
        plan.validate_settings(
            1,
            4 * 1024 * 1024,
            1,
            None,
            require_action_indices=True,
        )


def test_geometry_only_replay_plan_omits_actor_indices():
    wrapper = _wrapper()
    batch = _rollout(wrapper, batch=4, prompt=5, new_tokens=3)
    plan = build_replay_plan(
        batch,
        2,
        4 * 1024 * 1024,
        include_action_indices=False,
    )
    assert not plan.has_action_indices
    assert all(
        not shard.emit_index.numel() for shard in plan.shards()
    )
    plan.validate_settings(
        2,
        4 * 1024 * 1024,
        1,
        None,
        require_action_indices=False,
    )
    with pytest.raises(ValueError, match="action indices"):
        plan.validate_settings(
            2,
            4 * 1024 * 1024,
            1,
            None,
            require_action_indices=True,
        )


def test_replay_plan_preserves_empty_batch_behavior():
    batch = _rollout(_wrapper(), batch=2, prompt=5, new_tokens=3)
    empty = copy.copy(batch)
    for name, value in vars(batch).items():
        if isinstance(value, torch.Tensor) and value.dim() >= 1:
            setattr(empty, name, value[:0])
    empty.replay_layout_token = object()

    plan = build_replay_plan(empty, 2, 4 * 1024 * 1024)
    assert plan.batch_rows == 0
    assert not plan.specs
    assert plan.indices.dtype == torch.long
    assert plan.indices.numel() == 0
    assert list(iter_planned_replay_microbatches(empty, plan)) == []


def test_clip_bounds_are_the_same_constants_without_the_host_copy():
    # new_full replaced new_tensor purely to drop a blocking copy; the bound
    # it builds has to be the identical float.
    for dtype in (torch.float32, torch.bfloat16):
        reference = torch.zeros(3, dtype=dtype)
        for offset in (1.0 - 0.20, 1.0 + 0.28, 1.0 - 0.03):
            assert torch.equal(
                torch.log(reference.new_tensor(offset)),
                torch.log(reference.new_full((), offset)),
            )


def test_resume_schema_compatibility_is_strict_equality():
    """v28 has no migrations: every pre-hidden-carry schema is refused."""
    lockstep = execution_schema_for_rollout_scheduler("lockstep")
    refill = execution_schema_for_rollout_scheduler("continuous_refill")
    assert lockstep == EXECUTION_SCHEMA
    assert refill != lockstep
    assert resume_execution_schema_compatible({"execution_schema": lockstep})
    assert resume_execution_schema_compatible(
        {"execution_schema": refill},
        expected_execution_schema=refill,
    )
    assert not resume_execution_schema_compatible(
        {"execution_schema": refill}
    )
    for stale in (
        None,
        "unique_prefix_compact_tail_shuffled_pool1024_disjoint_b256_"
        "one_way_stop_reverse_kl_token_clip_anchored_value_general_lr_"
        "sequential_data/v27",
    ):
        assert not resume_execution_schema_compatible(
            {"execution_schema": stale}
        )
    assert resume_replay_schema_compatible(
        {"replay_numerics_schema": REPLAY_NUMERICS_SCHEMA}
    )
    assert not resume_replay_schema_compatible(
        {"replay_numerics_schema": "per_position_dense/v1"}
    )
    assert not resume_replay_schema_compatible({})
    with pytest.raises(ValueError, match="unknown rollout scheduler"):
        execution_schema_for_rollout_scheduler("tail_merge")
