from __future__ import annotations

import copy
import math
from types import MethodType, SimpleNamespace

import pytest
import torch

import train_gpt as baseline
from fresh_lejepa_train import FreshLeJEPAGPT
from fresh_lejepa_train_v1_probe_shared_rms_pope import FreshLeJEPASharedRMSV1PoPE
from postraining.core import (
    clipped_policy_loss,
    generalized_advantage_estimate,
    nearby_numeric_reward,
)
from postraining.latent_rollout import (
    PAD_SLOT,
    THOUGHT_SLOT,
    TOKEN_SLOT,
    assemble_stream_latents,
    assign_terminal_rewards,
    pack_rollout_groups_for_replay,
    continuation_reward,
    emitted_token_rows,
    half_forced_group_members,
    iter_length_aware_microbatches,
    refresh_old_statistics,
    replay_beliefs,
    replay_head_inputs,
    scatter_replay_statistics,
    select_trajectory_rows,
    select_thought_actions,
    rollout_continuations,
    split_rollout_groups,
    trim_stream,
)
from postraining.latent_thought import (
    EMIT,
    THINK,
    THOUGHT_DISTRIBUTION_SCHEMA,
    THOUGHT_INPUT_SCHEMA,
    THOUGHT_MEAN_SCHEMA,
    LatentThoughtModel,
)
from postraining.model_io import _pope_construction
from postraining.train_latent_vapo import (
    EXECUTION_SCHEMA,
    GAIN_SCALED_EXECUTION_SCHEMA,
    GAIN_SCALED_THOUGHT_INPUT_SCHEMA,
    PERFORMANCE_COMPATIBLE_EXECUTION_SCHEMA,
    PREVIOUS_EXECUTION_SCHEMA,
    MathPromptSampler,
    REWARD_SCHEMA,
    build_optimizers,
    evaluate_aime_latent,
    joint_action_logprobs,
    per_dimension_thought_policy_loss,
    math_dataset_identity,
    measure_post_update_policy_drift,
    migrate_zero_adapter_resume,
    resume_execution_schema_compatible,
    sampled_reverse_kl,
    sample_prompt_batch,
    score_math_rollout,
    think_run_lengths,
    rollout_diagnostics,
    save_checkpoint,
    update_minibatch,
    verify_terminated_answer,
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
        force_initial_think=torch.arange(batch).remainder(2) == 0,
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


def test_half_forced_group_members_is_exact_and_alternating_per_group():
    mask = half_forced_group_members(3, 4, torch.device("cpu"))
    assert mask.tolist() == [True, False, True, False] * 3
    assert torch.equal(mask.reshape(3, 4).sum(1), torch.full((3,), 2))
    with pytest.raises(ValueError, match="even"):
        half_forced_group_members(1, 3, torch.device("cpu"))


def test_rollout_emits_exactly_the_requested_tokens():
    wrapper = _wrapper()
    batch = _rollout(wrapper, batch=3, prompt=4, new_tokens=5)
    for row in emitted_token_rows(batch):
        assert len(row) == 5


def test_emit_only_rollout_still_runs_the_thought_mean_densely():
    wrapper = _wrapper()
    with torch.no_grad():
        wrapper.gate.head.weight.zero_()
        wrapper.gate.head.bias.fill_(100.0)
    projected_shapes = []
    handle = wrapper.transition.mean_head.register_forward_pre_hook(
        lambda _module, inputs: projected_shapes.append(tuple(inputs[0].shape))
    )
    try:
        batch = _rollout(wrapper, batch=3, prompt=4, new_tokens=5)
    finally:
        handle.remove()
    assert all(len(row) == 5 for row in emitted_token_rows(batch))
    assert projected_shapes
    assert all(
        shape == (3, 1, wrapper.backbone.tok_emb.embedding_dim)
        for shape in projected_shapes
    )


def test_thinking_is_unbounded_and_stream_budget_truncates():
    wrapper = _wrapper()
    # Bias the gate hard toward THINK: no watchdog interrupts, so runs grow
    # past the old 4-think cap and rows exhaust the stream budget instead of
    # being forced to emit.
    with torch.no_grad():
        wrapper.gate.head.bias.fill_(-5.0)
    stream_steps = 24
    batch = _rollout(
        wrapper, batch=2, prompt=4, new_tokens=3, stream_steps=stream_steps
    )
    generated = batch.kind[:, batch.prompt_length:]
    assert int(generated.size(1)) <= stream_steps
    longest_run = 0
    for row in range(generated.size(0)):
        run = 0
        for slot in generated[row].tolist():
            run = run + 1 if slot == THOUGHT_SLOT else 0
            longest_run = max(longest_run, run)
    assert longest_run > 4
    # Truncated rows emit fewer than the requested tokens and never more.
    rows = emitted_token_rows(batch)
    assert all(len(row) <= 3 for row in rows)
    assert any(len(row) < 3 for row in rows)


def test_stream_budget_smaller_than_emit_cap_is_rejected():
    wrapper = _wrapper()
    with pytest.raises(ValueError):
        rollout_continuations(wrapper, torch.randint(0, 32, (1, 4)), 8, 4, 1.0, 1.0)


def test_emit_only_policy_needs_and_uses_one_extra_stream_slot():
    wrapper = _deterministic_wrapper()
    prompt_ids = torch.randint(0, 32, (2, 4))
    unforced = rollout_continuations(
        wrapper, prompt_ids, 4, 4, 1.0, 1e-6,
        force_initial_think=False,
    )
    assert all(len(row) == 4 for row in emitted_token_rows(unforced))
    with pytest.raises(ValueError, match="forced initial thought"):
        rollout_continuations(
            wrapper, prompt_ids, 4, 4, 1.0, 1e-6,
            force_initial_think=True,
        )
    batch = rollout_continuations(
        wrapper, prompt_ids, 4, 5, 1.0, 1e-6,
        force_initial_think=True,
    )
    assert all(len(row) == 4 for row in emitted_token_rows(batch))
    assert torch.all(batch.action_mask.sum(1) == 5)
    assert torch.all(batch.gate_mask.sum(1) == 4)


def test_stream_storage_is_internally_consistent():
    wrapper = _wrapper()
    batch = _rollout(wrapper, batch=4)
    prompt = batch.prompt_length
    forced = torch.tensor([True, False, True, False])
    assert torch.all(batch.kind[:, :prompt] == TOKEN_SLOT)
    # The full-prompt belief always produces one latent action.  It is a
    # temporal/thought action but not a Bernoulli gate decision.
    assert torch.all(batch.action_mask[:, prompt - 1] == 1)
    assert torch.all(batch.gate_actions[forced, prompt - 1] == THINK)
    assert torch.all(batch.gate_mask[forced, prompt - 1] == 0)
    assert torch.all(batch.kind[forced, prompt] == THOUGHT_SLOT)
    assert torch.all(batch.gate_mask[~forced, prompt - 1] == 1)
    assert torch.all(batch.gate_mask <= batch.action_mask)
    assert torch.equal(
        batch.action_mask - batch.gate_mask,
        forced[:, None].float()
        * torch.nn.functional.one_hot(
            torch.tensor(prompt - 1), num_classes=batch.stream_length
        ).float(),
    )
    # Every recorded action produced a next input slot of a matching kind.
    action_positions = batch.action_mask.nonzero()
    for row, position in action_positions.tolist():
        action = int(batch.gate_actions[row, position])
        next_kind = int(batch.kind[row, position + 1])
        assert next_kind == (TOKEN_SLOT if action == EMIT else THOUGHT_SLOT)
    # PAD slots carry no thoughts, no actions, no rewards.
    pads = batch.kind == PAD_SLOT
    assert float(batch.action_mask[pads].sum()) == 0.0
    assert float(batch.gate_mask[pads].sum()) == 0.0
    assert float(batch.thoughts[pads].abs().sum()) == 0.0


@pytest.mark.parametrize(("gate_bias", "unforced_action"), ((100.0, EMIT), (-100.0, THINK)))
def test_forced_half_overrides_only_its_members_first_gate(
    gate_bias: float, unforced_action: int
):
    wrapper = _wrapper()
    with torch.no_grad():
        wrapper.gate.head.weight.zero_()
        wrapper.gate.head.bias.fill_(gate_bias)
    prompt_ids = torch.randint(0, 32, (4, 5))
    forced = half_forced_group_members(1, 4, torch.device("cpu"))
    batch = rollout_continuations(
        wrapper,
        prompt_ids,
        max_new_tokens=1,
        max_stream_steps=2,
        temperature=1.0,
        top_p=1.0,
        force_initial_think=forced,
    )
    boundary = batch.prompt_length - 1
    assert torch.all(batch.gate_actions[forced, boundary] == THINK)
    assert torch.all(batch.gate_actions[~forced, boundary] == unforced_action)
    assert torch.all(batch.gate_mask[forced, boundary] == 0)
    assert torch.all(batch.gate_mask[~forced, boundary] == 1)


def test_replay_reproduces_rollout_logprobs():
    wrapper = _wrapper()
    batch = _rollout(wrapper, batch=2, prompt=6, new_tokens=4)
    backbone = wrapper.backbone
    with torch.no_grad():
        stream_inputs, beliefs = replay_beliefs(wrapper, batch)
        features = wrapper.renderer_features(stream_inputs, beliefs)
        gate_logprobs = wrapper.gate.log_prob(batch.gate_actions.float(), beliefs)
        logits = backbone.logits_from_features(features).float()
        token_targets = torch.zeros_like(batch.token_ids)
        token_targets[:, :-1] = batch.token_ids[:, 1:]
        token_logprobs = logits.log_softmax(-1).gather(
            -1, token_targets[..., None]
        ).squeeze(-1)
    mask = batch.gate_mask.bool()
    # The rollout never values: old_values stay zero until the separate
    # critic fills them in refresh_old_statistics.
    assert float(batch.old_values.abs().sum()) == 0.0
    torch.testing.assert_close(
        gate_logprobs[mask], batch.old_gate_logprobs[mask], rtol=2e-4, atol=2e-4
    )
    emits = batch.emit_mask.bool()
    torch.testing.assert_close(
        token_logprobs[emits], batch.old_token_logprobs[emits], rtol=2e-4, atol=2e-4
    )


def test_assembled_latents_zero_pads_and_route_thoughts_through_adapter():
    wrapper = _wrapper()
    batch = _rollout(wrapper)
    with torch.no_grad():
        latents = assemble_stream_latents(wrapper, batch)
    pads = batch.kind == PAD_SLOT
    assert float(latents[pads].abs().sum()) == 0.0
    thought_slots = batch.kind == THOUGHT_SLOT
    if thought_slots.any():
        expected = wrapper.adapter(batch.thoughts).to(latents.dtype)
        torch.testing.assert_close(latents[thought_slots], expected[thought_slots])
    token_slots = batch.kind == TOKEN_SLOT
    expected_tokens = wrapper.embed_tokens(batch.token_ids)
    torch.testing.assert_close(latents[token_slots], expected_tokens[token_slots])


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


def test_forced_initial_think_and_first_emit_both_receive_terminal_credit():
    wrapper = _deterministic_wrapper()
    prompt_ids = torch.randint(0, 32, (2, 4))
    batch = trim_stream(
        rollout_continuations(
            wrapper,
            prompt_ids,
            max_new_tokens=1,
            max_stream_steps=2,
            temperature=1.0,
            top_p=1e-6,
            stop_ids=tuple(range(32)),
            force_initial_think=True,
        )
    )
    assert torch.all(batch.action_mask.sum(1) == 2)
    assert torch.all(batch.gate_mask.sum(1) == 1)
    scores = torch.tensor([0.25, 0.75])
    assign_terminal_rewards(batch, scores)
    advantages, _ = generalized_advantage_estimate(
        batch.rewards,
        batch.old_values,
        batch.action_mask,
        torch.ones(2),
    )
    for row, score in enumerate(scores):
        torch.testing.assert_close(
            advantages[row][batch.action_mask[row].bool()],
            torch.full((2,), float(score)),
        )


def test_joint_action_logprobs_match_emit_optional_and_forced_think() -> None:
    new, old = joint_action_logprobs(
        new_gate_logprobs=torch.tensor([[10.0, 20.0, 30.0]]),
        old_gate_logprobs=torch.tensor([[1.0, 2.0, 3.0]]),
        new_token_logprobs=torch.tensor([[100.0, 200.0, 300.0]]),
        old_token_logprobs=torch.tensor([[4.0, 5.0, 6.0]]),
        new_thought_logprobs=torch.tensor([[0.0, 2_000.0, 3_000.0]]),
        old_thought_logprobs=torch.tensor([[0.0, 7.0, 8.0]]),
        gate_mask=torch.tensor([[1.0, 1.0, 0.0]]),
        emit_mask=torch.tensor([[1.0, 0.0, 0.0]]),
    )
    # EMIT=gate+token; optional THINK=gate+thought; forced THINK=thought only.
    torch.testing.assert_close(new, torch.tensor([[110.0, 2_020.0, 3_000.0]]))
    torch.testing.assert_close(old, torch.tensor([[5.0, 9.0, 8.0]]))


def test_sampled_reverse_kl_matches_k3_value_and_gradient() -> None:
    new = torch.tensor([[0.2, -0.4], [0.0, 0.1]], requires_grad=True)
    old = torch.zeros_like(new)
    factors = sampled_reverse_kl(new, old)
    expected = torch.expm1(new.detach()) - new.detach()
    torch.testing.assert_close(factors, expected)

    coefficient = 0.3
    action_denominator = 5.0
    penalty = coefficient * factors.sum() / action_denominator
    penalty.backward()
    torch.testing.assert_close(
        new.grad,
        coefficient * torch.expm1(new.detach()) / action_denominator,
    )


def test_thought_policy_clips_dimensions_without_joint_ratio_coupling() -> None:
    # The product 1.2 * 1.2 exceeds the upper clip, but each latent dimension
    # is individually inside it. Per-dimension clipping therefore preserves
    # the mean factor ratio instead of clipping their joint product.
    new_thought = torch.log(torch.tensor([[1.2, 1.2]]))
    loss, clip_fraction, gate_clip_fraction = per_dimension_thought_policy_loss(
        new_gate_logprobs=torch.zeros(1),
        old_gate_logprobs=torch.zeros(1),
        new_thought_logprobs=new_thought,
        old_thought_logprobs=torch.zeros_like(new_thought),
        advantages=torch.ones(1),
        gate_mask=torch.zeros(1),
        action_denominator=torch.tensor(1.0),
    )
    torch.testing.assert_close(loss, torch.tensor(-1.2))
    torch.testing.assert_close(clip_fraction, torch.tensor(0.0))
    torch.testing.assert_close(gate_clip_fraction, torch.tensor(0.0))


def test_thought_policy_averages_dimension_credit_and_clip_fraction() -> None:
    new_thought = torch.log(torch.tensor([[1.5, 0.9]]))
    loss, clip_fraction, gate_clip_fraction = per_dimension_thought_policy_loss(
        new_gate_logprobs=torch.zeros(1),
        old_gate_logprobs=torch.zeros(1),
        new_thought_logprobs=new_thought,
        old_thought_logprobs=torch.zeros_like(new_thought),
        advantages=torch.ones(1),
        gate_mask=torch.zeros(1),
        action_denominator=torch.tensor(1.0),
    )
    torch.testing.assert_close(loss, torch.tensor(-(1.28 + 0.9) / 2))
    torch.testing.assert_close(clip_fraction, torch.tensor(0.5))
    torch.testing.assert_close(gate_clip_fraction, torch.tensor(0.0))


def test_optional_think_shared_gate_receives_one_action_gradient() -> None:
    gate = torch.zeros(1, requires_grad=True)
    dimensions = torch.zeros(1, 4, requires_grad=True)
    loss, clip_fraction, gate_clip_fraction = per_dimension_thought_policy_loss(
        new_gate_logprobs=gate,
        old_gate_logprobs=torch.zeros_like(gate),
        new_thought_logprobs=dimensions,
        old_thought_logprobs=torch.zeros_like(dimensions),
        advantages=torch.ones(1),
        gate_mask=torch.ones(1),
        action_denominator=torch.tensor(1.0),
    )
    loss.backward()
    torch.testing.assert_close(gate.grad, torch.tensor([-1.0]))
    torch.testing.assert_close(dimensions.grad, -torch.ones_like(dimensions))
    torch.testing.assert_close(loss, torch.tensor(-1.0))
    torch.testing.assert_close(clip_fraction, torch.tensor(0.0))
    torch.testing.assert_close(gate_clip_fraction, torch.tensor(0.0))


def test_forced_think_excludes_gate_from_factorwise_policy_gradient() -> None:
    gate = torch.zeros(1, requires_grad=True)
    dimensions = torch.zeros(1, 3, requires_grad=True)
    loss, _, gate_clip_fraction = per_dimension_thought_policy_loss(
        new_gate_logprobs=gate,
        old_gate_logprobs=torch.zeros_like(gate),
        new_thought_logprobs=dimensions,
        old_thought_logprobs=torch.zeros_like(dimensions),
        advantages=torch.ones(1),
        gate_mask=torch.zeros(1),
        action_denominator=torch.tensor(1.0),
    )
    loss.backward()
    torch.testing.assert_close(gate.grad, torch.zeros_like(gate))
    torch.testing.assert_close(dimensions.grad, -torch.ones_like(dimensions))
    torch.testing.assert_close(gate_clip_fraction, torch.tensor(0.0))


def test_forced_initial_think_trains_content_but_not_the_gate():
    wrapper = _wrapper()
    critic = _critic()
    batch = rollout_continuations(
        wrapper,
        torch.randint(0, 32, (4, 5)),
        max_new_tokens=0,
        max_stream_steps=1,
        temperature=1.0,
        top_p=1.0,
        generator=torch.Generator().manual_seed(17),
        force_initial_think=True,
    )
    assert batch.gate_mask.sum() == 0
    assign_terminal_rewards(batch, torch.tensor([0.1, 0.3, 0.6, 0.9]))
    refresh_old_statistics(wrapper, critic, batch)
    gate_before = [parameter.detach().clone() for parameter in wrapper.gate.parameters()]
    mean_before = [
        parameter.detach().clone()
        for parameter in wrapper.transition.mean_head.parameters()
    ]
    metrics = update_minibatch(
        wrapper, critic, batch, _optimizers(wrapper, critic)
    )
    assert torch.isfinite(torch.tensor(metrics["policy_loss"]))
    assert metrics["think_action_count"] == 0.0
    assert metrics["forced_initial_think_action_count"] == 4.0
    assert all(
        torch.equal(before, after)
        for before, after in zip(gate_before, wrapper.gate.parameters(), strict=True)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            mean_before,
            wrapper.transition.mean_head.parameters(),
            strict=True,
        )
    )


def test_diagnostics_separate_forced_initial_and_optional_thinking():
    wrapper = _deterministic_wrapper()
    batch = _rollout(wrapper, batch=4, prompt=5, new_tokens=3)
    batch.reward_scalar.copy_(torch.tensor([0.0, 0.25, 0.5, 1.0]))
    metrics = rollout_diagnostics(batch, samples_per_prompt=4)
    assert metrics["think_fraction"] == 0.0
    assert metrics["exact_accuracy"] == 0.25
    assert metrics["exact_within_group_reward_std"] == pytest.approx(
        3**0.5 / 4
    )
    assert metrics["partial_reward_fraction"] == 0.5
    assert metrics["forced_initial_thinks_per_trajectory"] == 0.5
    assert metrics["thoughts_per_trajectory"] == 0.5
    assert metrics["forced_initial_trajectory_fraction"] == 0.5
    assert metrics["reward_mean_forced_initial"] == pytest.approx(0.25)
    assert metrics["reward_mean_unforced_initial"] == pytest.approx(0.625)
    assert metrics["optional_thinking_trajectory_fraction"] == 0.0
    assert metrics["reward_mean_no_optional_thinking"] == pytest.approx(0.4375)


def test_gate_pg_coef_zero_freezes_the_gate_but_not_the_rest():
    wrapper = _wrapper()
    critic = _critic()
    with torch.no_grad():
        wrapper.backbone.policy_probe.output.weight.normal_(std=0.02)
        # Nonzero gate weights so an unfrozen update would visibly move them.
        wrapper.gate.head.weight.normal_(std=0.02)
    batch = _rollout(wrapper, batch=4, prompt=5, new_tokens=3)
    assign_terminal_rewards(batch, torch.rand(4))
    refresh_old_statistics(wrapper, critic, batch)
    gate_weight = wrapper.gate.head.weight.clone()
    gate_bias = wrapper.gate.head.bias.clone()
    trunk_before = wrapper.backbone.blocks[0].attn.proj.weight.clone()
    metrics = update_minibatch(
        wrapper,
        critic,
        batch,
        _optimizers(wrapper, critic),
        gate_pg_coef=0.0,
        gate_entropy_coef=0.2,
    )
    assert torch.equal(wrapper.gate.head.weight, gate_weight)
    assert torch.equal(wrapper.gate.head.bias, gate_bias)
    # Freezing the gate must not freeze the actor: the trunk still trains
    # through the renderer/thought terms.
    assert not torch.equal(wrapper.backbone.blocks[0].attn.proj.weight, trunk_before)
    assert torch.isfinite(torch.tensor(metrics["policy_loss"]))
    assert metrics["gate_entropy_bonus"] == 0.0


def test_gate_entropy_bonus_has_the_right_normalization():
    wrapper = _wrapper()
    critic = _critic()
    batch = _rollout(wrapper, batch=4, prompt=5, new_tokens=3)
    assign_terminal_rewards(batch, torch.rand(4))
    refresh_old_statistics(wrapper, critic, batch)

    metrics = update_minibatch(
        wrapper,
        critic,
        batch,
        _optimizers(wrapper, critic),
        gate_entropy_coef=0.2,
    )

    assert metrics["gate_entropy_coef"] == 0.2
    assert metrics["gate_entropy_bonus"] == pytest.approx(
        0.2 * metrics["gate_entropy"]
    )


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
    gate_before = wrapper.gate.head.weight.clone()
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
    assert not torch.equal(wrapper.gate.head.weight, gate_before)
    # Only the unused backbone critic probe stays frozen.
    assert torch.equal(backbone.critic_probe.output.weight, frozen_probe_before)
    assert metrics["trunk_grad_norm"] > 0.0


def test_refresh_old_statistics_matches_the_update_code_path_exactly():
    wrapper = _wrapper()
    critic = _critic()
    with torch.no_grad():
        # Exercise actual state- and dimension-dependent noise. Zero-init
        # would let this test pass even if refresh priced the wrong belief.
        wrapper.transition.log_sigma_head.weight.normal_(std=0.02)
        wrapper.transition.log_sigma_head.bias.copy_(
            torch.linspace(-2.4, -1.6, wrapper.backbone.tok_emb.embedding_dim)
        )
    batch = _rollout(wrapper, batch=2, prompt=6, new_tokens=4)
    refresh_old_statistics(wrapper, critic, batch)
    backbone = wrapper.backbone
    with torch.no_grad():
        beliefs, predicted, stream_inputs, token_targets = replay_head_inputs(
            wrapper, batch
        )
        values = critic.values(batch).float()
        gate_logprobs = wrapper.gate.log_prob(batch.gate_actions.float(), beliefs)
        emit_mask = batch.emit_mask.bool()
        emit_features = wrapper.renderer_features(
            stream_inputs[emit_mask], beliefs[emit_mask]
        )
        compact_token_logprobs = (
            backbone.logits_from_features(emit_features)
            .float()
            .log_softmax(-1)
            .gather(-1, token_targets[emit_mask][..., None])
            .squeeze(-1)
        )
        token_logprobs = torch.zeros_like(batch.old_token_logprobs)
        token_logprobs[emit_mask] = compact_token_logprobs
    assert torch.equal(batch.old_values, values)
    assert torch.equal(
        batch.old_gate_logprobs,
        gate_logprobs.float() * batch.gate_mask,
    )
    assert torch.equal(batch.old_token_logprobs, token_logprobs)
    with torch.no_grad():
        thought_means, thought_targets, think_mask = select_thought_actions(
            batch, predicted
        )
        compact_logprobs = wrapper.transition.per_dim_log_prob(
            thought_targets,
            thought_means,
            wrapper.transition.predict_log_sigma(beliefs[think_mask]),
        )
        thought_logprobs = torch.zeros_like(batch.old_thought_logprobs)
        thought_logprobs[think_mask] = compact_logprobs
    # Epoch-0 thought log-probability factors are identical by construction.
    assert torch.equal(batch.old_thought_logprobs, thought_logprobs.float())


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
        "old_gate_logprobs": batch.gate_mask.bool(),
        "old_token_logprobs": batch.emit_mask.bool(),
        "old_thought_logprobs": (
            (batch.gate_actions == THINK) & batch.action_mask.bool()
        ),
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
        force_initial_think=torch.tensor([True, False, True, False, False]),
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
    planned_rows = [row for _, rows, _ in shards for row in rows.tolist()]
    assert planned_rows == [3, 1, 2, 4, 0]
    for microbatch, rows, stream_length in shards:
        assert microbatch.kind.shape == (rows.numel(), stream_length)
        assert rows.numel() == 1 or rows.numel() * stream_length**2 <= budget


def test_length_aware_refresh_writes_noncontiguous_parent_rows():
    wrapper = _wrapper()
    critic = _critic()
    batch = _rollout(wrapper, batch=6, prompt=6, new_tokens=4)
    batch.old_values.fill_(float("nan"))
    batch.old_gate_logprobs.fill_(float("nan"))
    batch.old_token_logprobs.fill_(float("nan"))
    refresh_old_statistics(
        wrapper,
        critic,
        batch,
        max_trajectories=2,
        attention_budget=10**9,
    )
    assert torch.isfinite(batch.old_values[batch.action_mask.bool()]).all()
    assert torch.isfinite(batch.old_gate_logprobs[batch.gate_mask.bool()]).all()
    assert torch.isfinite(batch.old_token_logprobs[batch.emit_mask.bool()]).all()
    think_mask = (batch.gate_actions == THINK) & batch.action_mask.bool()
    assert torch.isfinite(batch.old_thought_logprobs[think_mask]).all()


def test_trajectory_microbatch_update_matches_full_group_objective_and_step():
    base_wrapper = _wrapper()
    with torch.no_grad():
        # Ensure optional THINK actions as well as the forced action are
        # represented on both sides of the trajectory-microbatch boundary.
        base_wrapper.gate.head.bias.fill_(-0.5)
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
    assert bool(((batch.gate_actions == THINK) & batch.gate_mask.bool()).any())
    # The stored behavior statistics stay frozen while all three actor
    # factors move. This makes the shard-invariance check exercise nonzero
    # KL accumulation rather than the trivial refresh-equals-current case.
    with torch.no_grad():
        base_wrapper.gate.head.bias.add_(0.01)
        renderer_output = base_wrapper.backbone.policy_probe.output.weight
        renderer_output.add_(
            torch.linspace(
                -0.01, 0.01, renderer_output.numel()
            ).reshape_as(renderer_output)
        )
        thought_bias = base_wrapper.transition.mean_head.bias
        thought_bias.add_(
            torch.linspace(-0.01, 0.01, thought_bias.numel())
        )

    full_wrapper = copy.deepcopy(base_wrapper)
    micro_wrapper = copy.deepcopy(base_wrapper)
    full_critic = copy.deepcopy(base_critic)
    micro_critic = copy.deepcopy(base_critic)
    full_optimizers = _optimizers(full_wrapper, full_critic)
    micro_optimizers = _optimizers(micro_wrapper, micro_critic)

    kwargs = dict(
        positive_lm_weight=0.1,
        positive_reward_threshold=0.5,
        thought_pg_coef=0.7,
        thought_reverse_kl_coef=0.3,
        gate_pg_coef=0.8,
        gate_entropy_coef=0.02,
    )
    full_metrics = update_minibatch(
        full_wrapper,
        full_critic,
        copy.deepcopy(batch),
        full_optimizers,
        replay_max_trajectories=6,
        replay_attention_budget=10**9,
        **kwargs,
    )
    micro_metrics = update_minibatch(
        micro_wrapper,
        micro_critic,
        copy.deepcopy(batch),
        micro_optimizers,
        replay_max_trajectories=2,
        replay_attention_budget=10**9,
        **kwargs,
    )

    for key in (
        "value_loss",
        "policy_loss",
        "thought_reverse_kl_penalty",
        "positive_lm_loss",
        "gate_entropy_bonus",
        "advantage_mean",
        "advantage_std",
        "gate_behavior_kl",
        "renderer_behavior_kl",
        "thought_behavior_kl_joint",
        "thought_behavior_kl_per_dim",
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

    assert full_metrics["gate_behavior_kl"] > 0
    assert full_metrics["renderer_behavior_kl"] > 0
    assert full_metrics["thought_behavior_kl_joint"] > 0
    thought_count = (
        (batch.gate_actions == THINK).float() * batch.action_mask
    ).sum()
    expected_reverse_kl_penalty = (
        0.3
        * full_metrics["thought_behavior_kl_joint"]
        * thought_count
        / batch.action_mask.sum()
    )
    torch.testing.assert_close(
        torch.tensor(full_metrics["thought_reverse_kl_penalty"]),
        expected_reverse_kl_penalty,
    )
    torch.testing.assert_close(
        torch.tensor(full_metrics["thought_behavior_kl_joint"]),
        torch.tensor(full_metrics["thought_behavior_kl_per_dim"])
        * batch.old_thought_logprobs.size(-1),
    )
    gate_count = batch.gate_mask.sum()
    emit_count = batch.emit_mask.sum()
    expected_policy_kl = (
        full_metrics["gate_behavior_kl"] * gate_count
        + full_metrics["renderer_behavior_kl"] * emit_count
        + full_metrics["thought_behavior_kl_joint"] * thought_count
    ) / batch.action_mask.sum()
    torch.testing.assert_close(
        torch.tensor(full_metrics["policy_behavior_kl_per_action"]),
        expected_policy_kl,
    )

    # Adam must see one full-group update, not one update per replay shard.
    for optimizer in micro_optimizers.values():
        steps = {
            int(state["step"])
            for state in optimizer.state.values()
            if "step" in state
        }
        assert steps == {1}


def test_thought_pg_gradient_reaches_the_trunk_and_fresh_mean_head():
    # The thought surrogate alone must backprop through the fresh thought mean
    # into both that head and its belief-producing trunk.
    wrapper = _wrapper()
    critic = _critic()
    with torch.no_grad():
        # Bias the gate toward THINK so thought actions certainly occur.
        wrapper.gate.head.bias.fill_(-2.0)
    batch = _rollout(wrapper, batch=4, prompt=5, new_tokens=3)
    thinks = (batch.gate_actions == THINK).float() * batch.action_mask
    assert thinks.sum() > 0
    assign_terminal_rewards(batch, torch.rand(4))
    refresh_old_statistics(wrapper, critic, batch)

    beliefs, predicted, _, _ = replay_head_inputs(wrapper, batch)
    thought_means, thought_targets, think_mask = select_thought_actions(
        batch, predicted
    )
    new_logprobs = wrapper.transition.per_dim_log_prob(
        thought_targets,
        thought_means,
        wrapper.transition.predict_log_sigma(beliefs[think_mask]),
    )
    advantages, _ = generalized_advantage_estimate(
        batch.rewards, batch.old_values, batch.action_mask, torch.ones(4)
    )
    loss, _, _ = clipped_policy_loss(
        new_logprobs.sum(-1),
        batch.old_thought_logprobs[think_mask].sum(-1),
        advantages.detach()[think_mask],
        torch.ones_like(advantages[think_mask]),
    )
    loss.backward()
    proj_grad = wrapper.backbone.blocks[0].attn.proj.weight.grad
    assert proj_grad is not None
    assert float(proj_grad.abs().sum()) > 0.0
    mean_head_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in wrapper.transition.mean_head.parameters()
        if parameter.grad is not None
    )
    assert mean_head_grad > 0.0
    assert all(
        parameter.grad is None
        for parameter in wrapper.backbone.prediction_projector.parameters()
    )


def test_absent_thought_objective_leaves_mean_and_sigma_grad_none_despite_momentum(
):
    wrapper = _wrapper()
    critic = _critic()
    optimizers = _optimizers(wrapper, critic)
    mean_parameters = list(wrapper.transition.mean_head.parameters())
    sigma_parameters = list(wrapper.transition.log_sigma_head.parameters())

    # Seed Adam state so a spurious zero gradient would move either head
    # through stale momentum even with weight decay disabled.
    optimizers["actor"].zero_grad(set_to_none=True)
    mean_parameters[0].grad = torch.ones_like(mean_parameters[0])
    sigma_parameters[0].grad = torch.ones_like(sigma_parameters[0])
    optimizers["actor"].step()
    optimizers["actor"].zero_grad(set_to_none=True)

    with torch.no_grad():
        wrapper.gate.head.weight.zero_()
        wrapper.gate.head.bias.fill_(-2.0)
    batch = _rollout(wrapper, batch=4, prompt=5, new_tokens=3)
    think_mask = (batch.gate_actions == THINK) & batch.action_mask.bool()
    assert bool(think_mask.any())
    assign_terminal_rewards(batch, torch.rand(4))
    refresh_old_statistics(wrapper, critic, batch)
    with torch.no_grad():
        wrapper.transition.mean_head.bias.add_(0.01)
    before = [parameter.detach().clone() for parameter in mean_parameters]
    sigma_before = [parameter.detach().clone() for parameter in sigma_parameters]
    metrics = update_minibatch(
        wrapper,
        critic,
        batch,
        optimizers,
        thought_pg_coef=0.0,
    )

    assert metrics["thought_behavior_kl_joint"] > 0
    assert all(parameter.grad is None for parameter in mean_parameters)
    assert all(parameter.grad is None for parameter in sigma_parameters)
    for parameter, reference in zip(mean_parameters, before, strict=True):
        torch.testing.assert_close(parameter, reference)
    for parameter, reference in zip(sigma_parameters, sigma_before, strict=True):
        torch.testing.assert_close(parameter, reference)


def test_reverse_kl_alone_anchors_the_continuous_thought_policy():
    wrapper = _wrapper()
    critic = _critic()
    with torch.no_grad():
        wrapper.gate.head.weight.zero_()
        wrapper.gate.head.bias.fill_(-2.0)
    batch = _rollout(wrapper, batch=4, prompt=5, new_tokens=3)
    assert bool(((batch.gate_actions == THINK) & batch.action_mask.bool()).any())
    assign_terminal_rewards(batch, torch.rand(4))
    refresh_old_statistics(wrapper, critic, batch)
    with torch.no_grad():
        wrapper.transition.mean_head.bias.add_(0.02)
    before = [
        parameter.detach().clone()
        for parameter in wrapper.transition.mean_head.parameters()
    ]

    metrics = update_minibatch(
        wrapper,
        critic,
        batch,
        _optimizers(wrapper, critic),
        thought_pg_coef=0.0,
        thought_reverse_kl_coef=0.3,
        gate_pg_coef=0.0,
    )

    assert metrics["thought_reverse_kl_penalty"] > 0.0
    assert any(
        not torch.equal(parameter, reference)
        for parameter, reference in zip(
            wrapper.transition.mean_head.parameters(), before, strict=True
        )
    )


def test_renderer_reads_belief_without_training_the_thought_mean():
    wrapper = _wrapper()
    backbone = wrapper.backbone
    with torch.no_grad():
        backbone.policy_probe.output.weight.normal_(std=0.02)
    batch = _rollout(wrapper, batch=2, prompt=5, new_tokens=3)
    stream_inputs, beliefs = replay_beliefs(wrapper, batch)
    predicted = wrapper.thought_mean(beliefs)
    features = wrapper.renderer_features(stream_inputs, beliefs)

    torch.testing.assert_close(features[..., : beliefs.size(-1)], stream_inputs)
    torch.testing.assert_close(features[..., beliefs.size(-1) :], beliefs)
    assert not torch.equal(beliefs, predicted)

    backbone.zero_grad(set_to_none=True)
    backbone.logits_from_features(features).float().square().mean().backward()
    trunk_grad = backbone.blocks[0].attn.proj.weight.grad
    assert trunk_grad is not None
    assert float(trunk_grad.abs().sum()) > 0.0
    assert all(
        parameter.grad is None
        for parameter in wrapper.transition.mean_head.parameters()
    )


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
    # Six groups: trunk, Bernoulli gate, recurrent adapter, renderer probe,
    # learned log-sigma, and fresh thought mean. Every actor component uses
    # the same general learning rate as the critic.
    lrs = [group["lr"] for group in optimizers["actor"].param_groups]
    assert lrs == [1e-3] * 6
    assert optimizers["critic"].param_groups[0]["lr"] == 1e-3
    gate_group = optimizers["actor"].param_groups[1]["params"]
    assert {id(p) for p in gate_group} == {
        id(p) for p in wrapper.gate.parameters()
    }
    adapter_group = optimizers["actor"].param_groups[2]["params"]
    assert {id(p) for p in adapter_group} == {
        id(p) for p in wrapper.adapter.parameters()
    }
    renderer_group = optimizers["actor"].param_groups[3]["params"]
    assert {id(p) for p in renderer_group} == {
        id(p) for p in wrapper.backbone.policy_probe.parameters()
    }
    sigma_group = optimizers["actor"].param_groups[4]["params"]
    assert {id(p) for p in sigma_group} == {
        id(p) for p in wrapper.transition.log_sigma_head.parameters()
    }
    mean_group = optimizers["actor"].param_groups[5]["params"]
    assert {id(p) for p in mean_group} == {
        id(p) for p in wrapper.transition.mean_head.parameters()
    }


def test_post_update_drift_measures_the_deployed_policy_move():
    wrapper = _wrapper()
    critic = _critic()
    batch = _rollout(wrapper, batch=4, prompt=5, new_tokens=3)
    refresh_old_statistics(wrapper, critic, batch)
    grad_modes = []
    output_requires_grad = []
    diagnostic_replay_calls = 0

    def diagnostic_replay(*args, **kwargs):
        nonlocal diagnostic_replay_calls
        diagnostic_replay_calls += 1
        assert not torch.is_grad_enabled()
        return replay_head_inputs(*args, **kwargs)

    def record_grad_mode(_module, _inputs, output):
        grad_modes.append(torch.is_grad_enabled())
        output_requires_grad.append(output.requires_grad)

    handle = wrapper.transition.mean_head.register_forward_hook(
        record_grad_mode
    )
    try:
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
            wrapper.gate.head.bias.add_(0.2)

        drift = measure_post_update_policy_drift(
            wrapper,
            [batch],
            replay_max_trajectories=32,
            replay_attention_budget=4 * 1024 * 1024,
            replay_bucket=1,
            replay_function=diagnostic_replay,
        )
    finally:
        handle.remove()

    assert grad_modes and not any(grad_modes)
    assert output_requires_grad and not any(output_requires_grad)
    assert diagnostic_replay_calls == 2
    assert drift["kl/post_update_gate_behavior"] > 0.0
    assert drift["kl/post_update_policy_behavior_per_action"] > 0.0
    assert drift["ratio/post_update_joint_abs_log_max"] > 0.0


def test_post_update_kl_stays_finite_when_joint_thought_ratio_is_huge():
    wrapper = _wrapper()
    critic = _critic()
    with torch.no_grad():
        wrapper.gate.head.bias.fill_(-2.0)
    batch = _rollout(wrapper, batch=4, prompt=5, new_tokens=3)
    refresh_old_statistics(wrapper, critic, batch)
    think_mask = (
        (batch.gate_actions == THINK) & batch.action_mask.bool()
    )
    assert think_mask.any()
    # Every factor has a benign finite ratio, but their summed joint log-ratio
    # is 100, whose exponent overflows fp32 in the old joint-k3 diagnostic.
    latent_dim = batch.old_thought_logprobs.size(-1)
    batch.old_thought_logprobs[think_mask] -= 100.0 / latent_dim
    drift = measure_post_update_policy_drift(
        wrapper,
        [batch],
        replay_max_trajectories=32,
        replay_attention_budget=4 * 1024 * 1024,
        replay_bucket=1,
    )
    assert math.isfinite(drift["kl/post_update_policy_behavior_per_action"])
    assert drift["kl/post_update_policy_behavior_per_action"] > 0.0
    assert drift["ratio/post_update_joint_abs_log_max"] >= 99.0
    assert drift["ratio/post_update_thought_joint_abs_log_max"] >= 99.0


def test_actor_accumulation_defers_the_trunk_step_to_the_caller():
    # The trainer takes one accumulated actor step per PPO epoch: minibatch
    # calls run with actor_step=False (grads accumulate, actor params
    # untouched) and the caller steps once at epoch end.
    torch.manual_seed(41)
    wrapper = _wrapper()
    critic = _critic()
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
    # Zero-init probe/gate outputs would make the clip-fraction assertions
    # below vacuous (both code paths output exactly zero); randomize them so
    # the ratio-one property is actually load-bearing.
    with torch.no_grad():
        wrapper.backbone.policy_probe.output.weight.normal_(std=0.02)
        wrapper.gate.head.weight.normal_(std=0.02)
    batch = _rollout(wrapper, batch=2, prompt=5, new_tokens=3)
    assert batch.thoughts.dtype == torch.float32
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
    assert metrics["gate_behavior_kl"] == 0.0
    assert metrics["renderer_behavior_kl"] == 0.0
    assert metrics["thought_behavior_kl_joint"] == 0.0
    assert metrics["thought_behavior_kl_per_dim"] == 0.0
    assert metrics["policy_behavior_kl_per_action"] == 0.0
    assert metrics["joint_abs_log_ratio_max"] == 0.0
    assert metrics["harmful_positive_log_ratio_max"] == 0.0


def test_later_disjoint_minibatch_keeps_the_pool_behavior_policy_fixed():
    wrapper = _wrapper()
    critic = _critic()
    with torch.no_grad():
        wrapper.backbone.policy_probe.output.weight.normal_(std=0.02)
        wrapper.gate.head.weight.normal_(std=0.02)
    first = _rollout(wrapper, batch=4, prompt=5, new_tokens=3, seed=17)
    later = _rollout(wrapper, batch=4, prompt=5, new_tokens=3, seed=19)
    for batch in (first, later):
        assign_terminal_rewards(batch, torch.tensor([1.0, 0.0, 1.0, 0.0]))
        refresh_old_statistics(wrapper, critic, batch)
    frozen_later = {
        name: getattr(later, name).clone()
        for name in (
            "old_values",
            "old_gate_logprobs",
            "old_token_logprobs",
            "old_thought_logprobs",
        )
    }

    optimizers = _optimizers(wrapper, critic, learning_rate=1e-2)
    first_metrics = update_minibatch(
        wrapper,
        critic,
        first,
        optimizers,
        positive_lm_weight=1.0,
    )
    assert first_metrics["joint_abs_log_ratio_max"] == 0.0
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
    assert later_metrics["joint_abs_log_ratio_max"] > 0.0
    assert later_metrics["thought_joint_abs_log_ratio_max"] > 0.0
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
    assert metrics["prompt_groups"] == 2
    assert metrics["prompt_any_correct_fraction"] == 0.5
    assert metrics["prompt_mixed_reward_fraction"] == 0.0
    assert metrics["prompt_all_correct_fraction"] == 0.5
    assert metrics["prompt_zero_correct_fraction"] == 0.5
    assert metrics["within_group_reward_std"] == 0.0
    assert metrics["forced_initial_fraction"] == 0.5
    assert metrics["forced_initial_accuracy"] == 0.5
    assert metrics["unforced_initial_accuracy"] == 0.5
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
    assert 0.0 <= metrics["think_fraction"] <= 1.0
    # The eval must not perturb training RNG state.
    before = torch.get_rng_state()
    evaluate_aime_latent(
        wrapper, _Tokenizer(), rows[:1], samples=2, max_new_tokens=2,
        max_stream_steps=8, chunk=2, seed=5, device=torch.device("cpu"),
        prompt_tokens=8,
    )
    assert torch.equal(before, torch.get_rng_state())


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
    monkeypatch.setattr(
        evaluator,
        "emitted_token_and_kind_rows",
        lambda batch: (
            [[5] for _ in range(batch.kind.size(0))],
            batch.kind[:, batch.prompt_length :].tolist(),
        ),
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
    for problem in range(4):
        group = attempts[problem * 4 : (problem + 1) * 4]
        assert [attempt["forced_initial_think"] for attempt in group] == [
            True,
            False,
            True,
            False,
        ]
    for attempt in attempts:
        trace = str(attempt["action_trace"])
        assert attempt["terminated"] is True
        assert attempt["correct"] is True
        assert attempt["parsed_answer"] == "42"
        assert attempt["total_thought_count"] == trace.count("T")
        assert attempt["optional_thought_count"] == (
            trace.count("T") - int(bool(attempt["forced_initial_think"]))
        )
        assert sum(attempt["think_run_lengths"]) == trace.count("T")
        assert attempt["emitted_token_ids"] == [5]

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


def test_evaluate_aime_latent_preserves_half_member_assignment_across_chunks(
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

    seen = []
    original = evaluator.rollout_continuations

    def _spy(*args, **kwargs):
        seen.append(kwargs["force_initial_think"].tolist())
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
    assert seen == [[True, False, True], [False]]


def test_finished_row_compaction_preserves_original_row_attribution():
    wrapper = _deterministic_wrapper()
    prompt_ids = torch.randint(0, 32, (8, 4))
    forced = torch.arange(8).remainder(2) == 0

    def run(compact_finished: bool):
        torch.manual_seed(31)
        return rollout_continuations(
            wrapper,
            prompt_ids,
            max_new_tokens=0,
            max_stream_steps=1,
            temperature=1.0,
            top_p=1e-6,
            force_initial_think=forced,
            compact_finished=compact_finished,
        )

    compact = run(True)
    reference = run(False)
    for field in (
        "kind",
        "token_ids",
        "thoughts",
        "gate_actions",
        "action_mask",
        "gate_mask",
    ):
        torch.testing.assert_close(getattr(compact, field), getattr(reference, field))
    boundary = compact.prompt_length - 1
    assert compact.action_mask[:, boundary].bool().tolist() == forced.tolist()


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

    def staged_tokens(logits, _temperature, _top_p):
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
        max_stream_steps=32,
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

    def staged_tokens(logits, _temperature, _top_p):
        nonlocal calls
        tokens = torch.full(
            (logits.size(0),), 6, dtype=torch.long, device=logits.device
        )
        if calls == 0:
            tokens[:6] = 5
        calls += 1
        return tokens

    import postraining.latent_rollout as latent_rollout

    monkeypatch.setattr(latent_rollout, "top_p_sample", staged_tokens)
    rollout_continuations(
        wrapper,
        prompt_ids,
        max_new_tokens=32,
        max_stream_steps=32,
        temperature=1.0,
        top_p=0.7,
        stop_ids=(5,),
        compact_finished=True,
        finished_batch_size=16,
        record_likelihoods=False,
    )
    assert set(observed_batch_sizes) == {8}


def test_finished_row_compaction_preserves_model_dependent_survivor_tokens(
    monkeypatch,
):
    wrapper = _deterministic_wrapper()
    prompt_ids = torch.randint(1, 32, (128, 4))
    prompt_ids[:64, 0] = 0
    prompt_lengths = torch.tensor([3] * 64 + [4] * 64)
    calls = 0

    def terminate_then_argmax(logits, _temperature, _top_p):
        nonlocal calls
        if calls == 0:
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
            max_stream_steps=32,
            temperature=1.0,
            top_p=0.7,
            stop_ids=(5,),
            prompt_lengths=prompt_lengths,
            compact_finished=compact_finished,
            record_likelihoods=False,
        )

    fixed = emitted_token_rows(run(False))
    compacted = emitted_token_rows(run(True))
    assert compacted[:80] == [[5]] * 80
    assert compacted[80:] == fixed[80:]


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
                kwargs["force_initial_think"].clone(),
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
        forced,
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
    assert forced.tolist() == [True, False, True, False] * 3
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
        force_initial_think=torch.tensor([True, False]),
        tensor_positions=True,
        replay_storage=False,
    )
    assert batch.thoughts.shape == (2, 11, 0)
    assert batch.old_thought_logprobs.shape == (2, 11, 0)
    assert batch.action_mask.sum() >= 2
    wrapper.step_core = original


def test_eval_only_sampling_is_policy_identical_to_replay_rollout():
    wrapper = _wrapper()
    prompt_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    forced = torch.tensor([True, False])

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
            force_initial_think=forced,
            replay_storage=replay_storage,
        )

    replay = run(True)
    evaluation = run(False)
    for field in (
        "kind",
        "token_ids",
        "gate_actions",
        "action_mask",
        "gate_mask",
        "emit_mask",
    ):
        assert torch.equal(getattr(evaluation, field), getattr(replay, field))
    assert evaluation.thoughts.shape[-1] == 0
    assert replay.thoughts.shape[-1] == KWARGS["model_dim"]


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


def test_positive_lm_loss_applies_only_above_the_reward_threshold():
    wrapper = _wrapper()
    critic = _critic()
    backbone = wrapper.backbone
    batch = _rollout(wrapper, batch=4, prompt=5, new_tokens=3)
    optimizers = _optimizers(wrapper, critic, learning_rate=1e-4)
    assign_terminal_rewards(batch, torch.tensor([0.9, 0.1, 0.6, 0.2]))
    refresh_old_statistics(wrapper, critic, batch)
    metrics = update_minibatch(
        wrapper, critic, batch, optimizers, positive_lm_weight=0.1
    )
    assert metrics["positive_fraction"] == 0.5
    # NLL of real emitted tokens under a softcapped 32-way softmax is
    # strictly positive whenever any trajectory qualifies.
    assert metrics["positive_lm_loss"] > 0.0
    # No qualifying trajectory: the loss term is exactly zero.
    assign_terminal_rewards(batch, torch.tensor([0.1, 0.2, 0.3, 0.4]))
    refresh_old_statistics(wrapper, critic, batch)
    metrics = update_minibatch(
        wrapper, critic, batch, optimizers, positive_lm_weight=0.1
    )
    assert metrics["positive_fraction"] == 0.0
    assert metrics["positive_lm_loss"] == 0.0


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


def test_think_run_lengths_matches_hand_computation():
    T, H, P = TOKEN_SLOT, THOUGHT_SLOT, PAD_SLOT
    kind = torch.tensor(
        [
            # Runs of 2 and 1; a trailing think run cut off by padding.
            [T, T, H, H, T, H, P, P],
            # No thinks at all.
            [T, T, T, T, T, T, T, P],
            # Single run of 3 reaching the stream end.
            [T, T, T, T, T, H, H, H],
        ]
    )
    lengths = think_run_lengths(kind)
    assert sorted(lengths.tolist()) == [1.0, 2.0, 3.0]
    assert think_run_lengths(kind[1:2]).numel() == 0


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
        args=SimpleNamespace(value_warmup_steps=50),
        sampler=sampler,
        warmup_step=20,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["step"] == 0
    assert payload["value_warmup_step"] == 20
    assert payload["sampler_cursor"] == 3
    assert payload["execution_schema"] == EXECUTION_SCHEMA
    assert payload["reward_schema"] == REWARD_SCHEMA
    assert payload["thought_distribution_schema"] == THOUGHT_DISTRIBUTION_SCHEMA
    assert payload["thought_mean_schema"] == THOUGHT_MEAN_SCHEMA


def test_resume_schema_requires_matching_explicit_migration() -> None:
    current = {"execution_schema": EXECUTION_SCHEMA}
    assert resume_execution_schema_compatible(current)
    assert not resume_execution_schema_compatible(
        current,
        allow_reverse_kl_migration=True,
    )
    assert not resume_execution_schema_compatible(
        current,
        allow_performance_migration=True,
    )
    performance_previous = {
        "execution_schema": PERFORMANCE_COMPATIBLE_EXECUTION_SCHEMA
    }
    assert not resume_execution_schema_compatible(performance_previous)
    assert resume_execution_schema_compatible(
        performance_previous,
        allow_performance_migration=True,
    )
    assert not resume_execution_schema_compatible(
        performance_previous,
        allow_reverse_kl_migration=True,
        allow_performance_migration=True,
    )
    previous = {"execution_schema": PREVIOUS_EXECUTION_SCHEMA}
    assert not resume_execution_schema_compatible(previous)
    assert not resume_execution_schema_compatible(
        previous,
        allow_reverse_kl_migration=True,
    )
    assert resume_execution_schema_compatible(
        previous,
        allow_reverse_kl_migration=True,
        allow_performance_migration=True,
    )
    # Even trained v18 state is structurally resumable because v19 changes
    # only the objective; the explicit flag prevents an accidental change.
    assert not resume_execution_schema_compatible(
        {
            "execution_schema": PREVIOUS_EXECUTION_SCHEMA,
            "step": 1,
            "optimizers": {"actor": {"state": {1: {"step": 1}}}},
        }
    )
    assert not resume_execution_schema_compatible(
        {
            "execution_schema": PREVIOUS_EXECUTION_SCHEMA,
            "step": 1,
            "optimizers": {"actor": {"state": {1: {"step": 1}}}},
        },
        allow_reverse_kl_migration=True,
    )
    assert resume_execution_schema_compatible(
        {
            "execution_schema": PREVIOUS_EXECUTION_SCHEMA,
            "step": 1,
            "optimizers": {"actor": {"state": {1: {"step": 1}}}},
        },
        allow_reverse_kl_migration=True,
        allow_performance_migration=True,
    )

    # Older actor optimizers and policy semantics must fail at the schema
    # guard, not deep inside optimizer loading.
    legacy = {
        "execution_schema": "frozen_pool_2048_four_disjoint_b512_stable_actor_lrs/v6",
        "args": {
            "prompts_per_rollout": 16,
            "prompts_per_minibatch": 16,
            "samples_per_prompt": 32,
            "ppo_epochs": 1,
        },
    }
    assert not resume_execution_schema_compatible(legacy)
    assert not resume_execution_schema_compatible(
        {"execution_schema": "configurable_disjoint_b512_behavior_pool/v7"}
    )
    assert not resume_execution_schema_compatible(
        {"execution_schema": GAIN_SCALED_EXECUTION_SCHEMA}
    )
    assert not resume_execution_schema_compatible({"execution_schema": "older/v5"})


def test_zero_adapter_resume_migration_preserves_unrelated_adam_state() -> None:
    wrapper = _wrapper()
    critic = _critic()
    optimizer = _optimizers(wrapper, critic)["actor"]
    # Materialize a distinct moment tensor for every current actor parameter.
    for index, parameter in enumerate(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ):
        parameter.grad = torch.full_like(parameter, (index + 1) / 1000)
    optimizer.step()
    current_optimizer = copy.deepcopy(optimizer.state_dict())
    weight_id, bias_id = current_optimizer["param_groups"][2]["params"]
    scalar_id = max(
        parameter_id
        for group in current_optimizer["param_groups"]
        for parameter_id in group["params"]
    ) + 1
    current_optimizer["param_groups"][2]["params"] = [
        scalar_id, weight_id, bias_id
    ]
    current_optimizer["state"][scalar_id] = {
        "step": torch.tensor(7.0),
        "exp_avg": torch.tensor(0.25),
        "exp_avg_sq": torch.tensor(0.5),
    }
    unaffected_ids = {
        parameter_id
        for group_index, group in enumerate(current_optimizer["param_groups"])
        if group_index != 2
        for parameter_id in group["params"]
    }
    unaffected_before = {
        parameter_id: copy.deepcopy(current_optimizer["state"][parameter_id])
        for parameter_id in unaffected_ids
    }

    model = copy.deepcopy(wrapper.state_dict())
    model["adapter.projection.weight"].fill_(1.0)
    model["adapter.projection.bias"].fill_(2.0)
    model["adapter.interpolation_strength"] = torch.tensor(-4.5e-4)
    critic_state = {"sentinel": torch.arange(4)}
    critic_optimizer_state = {"sentinel": torch.arange(3)}
    cpu_rng_state = torch.random.get_rng_state().clone()
    payload = {
        "execution_schema": GAIN_SCALED_EXECUTION_SCHEMA,
        "thought_input_schema": GAIN_SCALED_THOUGHT_INPUT_SCHEMA,
        "model": model,
        "critic": critic_state,
        "optimizers": {
            "actor": current_optimizer,
            "critic": critic_optimizer_state,
        },
        "step": 640,
        "sampler_cursor": 11040,
        "cpu_rng": cpu_rng_state,
        "cuda_rng": [torch.arange(2, dtype=torch.uint8)],
        "python_rng": (3, (1, 2, 3), None),
    }

    provenance = migrate_zero_adapter_resume(payload, wrapper)

    assert payload["execution_schema"] == PREVIOUS_EXECUTION_SCHEMA
    assert payload["thought_input_schema"] == THOUGHT_INPUT_SCHEMA
    assert provenance["source_adapter_strength"] == pytest.approx(-4.5e-4)
    assert payload["step"] == 640
    assert payload["sampler_cursor"] == 11040
    assert payload["critic"] is critic_state
    assert payload["optimizers"]["critic"] is critic_optimizer_state
    torch.testing.assert_close(payload["cpu_rng"], cpu_rng_state)
    torch.testing.assert_close(payload["cuda_rng"][0], torch.arange(2, dtype=torch.uint8))
    assert payload["python_rng"] == (3, (1, 2, 3), None)
    assert "adapter.interpolation_strength" not in model
    assert torch.count_nonzero(model["adapter.projection.weight"]) == 0
    assert torch.count_nonzero(model["adapter.projection.bias"]) == 0
    assert current_optimizer["param_groups"][2]["params"] == [
        weight_id, bias_id
    ]
    assert all(
        parameter_id not in current_optimizer["state"]
        for parameter_id in (scalar_id, weight_id, bias_id)
    )
    for parameter_id, expected in unaffected_before.items():
        for key, value in expected.items():
            torch.testing.assert_close(
                current_optimizer["state"][parameter_id][key], value
            )

    migrated_wrapper = _wrapper(seed=99)
    migrated_wrapper.load_state_dict(model, strict=True)
    migrated_optimizer = _optimizers(migrated_wrapper, _critic(), 1e-3)["actor"]
    migrated_optimizer.load_state_dict(current_optimizer)
    assert not migrated_optimizer.state[
        migrated_wrapper.adapter.projection.weight
    ]
    assert not migrated_optimizer.state[
        migrated_wrapper.adapter.projection.bias
    ]


def test_zero_adapter_resume_migration_rejects_wrong_source_schema() -> None:
    wrapper = _wrapper()
    with pytest.raises(ValueError, match="requires execution schema"):
        migrate_zero_adapter_resume(
            {
                "execution_schema": EXECUTION_SCHEMA,
                "thought_input_schema": THOUGHT_INPUT_SCHEMA,
            },
            wrapper,
        )


@pytest.mark.parametrize(
    "malformation",
    ["missing_scalar", "bad_weight", "bad_groups", "duplicate_ids"],
)
def test_zero_adapter_resume_migration_rejects_malformed_layout_before_mutation(
    malformation: str,
) -> None:
    wrapper = _wrapper()
    critic = _critic()
    optimizer = _optimizers(wrapper, critic)["actor"].state_dict()
    weight_id, bias_id = optimizer["param_groups"][2]["params"]
    scalar_id = max(
        parameter_id
        for group in optimizer["param_groups"]
        for parameter_id in group["params"]
    ) + 1
    optimizer["param_groups"][2]["params"] = [scalar_id, weight_id, bias_id]
    model = copy.deepcopy(wrapper.state_dict())
    model["adapter.interpolation_strength"] = torch.tensor(1e-4)
    payload = {
        "execution_schema": GAIN_SCALED_EXECUTION_SCHEMA,
        "thought_input_schema": GAIN_SCALED_THOUGHT_INPUT_SCHEMA,
        "model": model,
        "optimizers": {"actor": optimizer},
    }
    if malformation == "missing_scalar":
        del model["adapter.interpolation_strength"]
    elif malformation == "bad_weight":
        model["adapter.projection.weight"] = torch.zeros(1)
    elif malformation == "bad_groups":
        optimizer["param_groups"][2]["params"] = [weight_id, bias_id]
    else:
        optimizer["param_groups"][2]["params"] = [scalar_id, weight_id, weight_id]
    before = copy.deepcopy(payload)

    with pytest.raises(ValueError):
        migrate_zero_adapter_resume(payload, wrapper)

    assert payload.keys() == before.keys()
    assert payload["execution_schema"] == before["execution_schema"]
    assert payload["thought_input_schema"] == before["thought_input_schema"]
    assert payload["model"].keys() == before["model"].keys()
    for key, value in payload["model"].items():
        torch.testing.assert_close(value, before["model"][key])


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
        torch.testing.assert_close(masked.predicted, reference.predicted)
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
        # Token sampling draws from the GLOBAL rng (only gate/thought use the
        # explicit generator), so every roll must restart it.
        torch.manual_seed(29)
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
        assert torch.equal(static.gate_actions, dynamic.gate_actions)
        assert torch.equal(static.action_mask, dynamic.action_mask)
        assert torch.equal(static.gate_mask, dynamic.gate_mask)
        torch.testing.assert_close(static.thoughts, dynamic.thoughts)
        torch.testing.assert_close(
            static.old_gate_logprobs, dynamic.old_gate_logprobs
        )
        torch.testing.assert_close(
            static.old_token_logprobs, dynamic.old_token_logprobs
        )


def test_preallocated_caches_that_do_not_fit_are_rejected():
    wrapper = _wrapper()
    device = torch.device("cpu")
    prompt_ids = torch.randint(0, 32, (2, 5))
    small = wrapper.make_static_generation_cache(2, 8, device)
    with pytest.raises(ValueError, match="do not fit"):
        rollout_continuations(
            wrapper, prompt_ids, 4, 12, 1.0, 1.0, caches=small
        )
    wrong_batch = wrapper.make_static_generation_cache(3, 32, device)
    with pytest.raises(ValueError, match="do not fit"):
        rollout_continuations(
            wrapper, prompt_ids, 4, 12, 1.0, 1.0, caches=wrong_batch
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
    for name in ("kind", "token_ids", "thoughts", "old_values"):
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
        "old_gate_logprobs",
        "old_token_logprobs",
        "old_thought_logprobs",
    ):
        stored = getattr(batch, name)
        assert not stored.requires_grad, name
        assert stored.grad_fn is None, name


def _deterministic_wrapper() -> LatentThoughtModel:
    """All-EMIT gate + argmax tokens make trajectories RNG-independent."""
    wrapper = _wrapper()
    with torch.no_grad():
        wrapper.gate.head.bias.fill_(40.0)
    # Forced latents still sample when every optional gate emits. Bypass only
    # the test draw so batched and sequential rows can consume different RNG
    # shapes without changing the actual thought; production sigma is bounded
    # away from zero by design.
    wrapper.transition.sample_latent = MethodType(
        lambda _self, mean, _log_sigma, generator=None: mean.float(),
        wrapper.transition,
    )
    return wrapper


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
        force_initial_think=half_forced_group_members(
            len(prompts), samples, torch.device("cpu")
        ),
    )
    groups = split_rollout_groups(batched, samples, prompt_lengths)

    for group, prompt in zip(groups, prompts, strict=True):
        torch.manual_seed(29)
        expected = trim_stream(
            rollout_continuations(
                wrapper, prompt[None].expand(samples, -1), 4, 8, 1.0, 1e-6,
                force_initial_think=half_forced_group_members(
                    1, samples, torch.device("cpu")
                ),
            )
        )
        # The padded rows shift every position by a constant, and PoPE
        # scores depend only on position differences — after undoing the
        # padding the trajectories must be identical.
        assert group.prompt_length == expected.prompt_length
        assert group.stream_length == expected.stream_length
        assert torch.equal(group.kind, expected.kind)
        assert torch.equal(group.token_ids, expected.token_ids)
        assert torch.equal(group.gate_actions, expected.gate_actions)
        assert torch.equal(group.action_mask, expected.action_mask)
        assert torch.equal(group.gate_mask, expected.gate_mask)
        assert torch.equal(group.emit_mask, expected.emit_mask)
        torch.testing.assert_close(group.thoughts, expected.thoughts)
        torch.testing.assert_close(
            group.old_token_logprobs,
            expected.old_token_logprobs,
            rtol=1e-4,
            atol=1e-5,
        )
        torch.testing.assert_close(
            group.old_gate_logprobs,
            expected.old_gate_logprobs,
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
    assert torch.equal(plain.gate_actions, padded.gate_actions)
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

    combined = pack_rollout_groups_for_replay(groups)
    assert combined.prompt_length == 4
    assert combined.stream_length == max(group.stream_length for group in groups)
    row_start = 0
    for group in groups:
        row_end = row_start + samples
        for name in (
            "kind",
            "token_ids",
            "thoughts",
            "gate_actions",
            "action_mask",
            "gate_mask",
            "emit_mask",
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
    assert metrics["joint_abs_log_ratio_max"] == 0.0
    assert metrics["policy_clip_fraction"] == 0.0

    combined.old_gate_logprobs.copy_(
        torch.arange(combined.old_gate_logprobs.numel()).reshape_as(
            combined.old_gate_logprobs
        )
    )
    combined.old_token_logprobs.copy_(combined.old_gate_logprobs + 1)
    combined.old_values.copy_(combined.old_gate_logprobs + 2)
    # Rollout storage starts with a zero-width Gaussian-stat field; refresh
    # expands it to one factor per latent dimension.
    combined.old_thought_logprobs = combined.thoughts + 3
    scatter_replay_statistics(combined, groups)
    repacked = pack_rollout_groups_for_replay(groups)
    for name in (
        "old_gate_logprobs",
        "old_token_logprobs",
        "old_thought_logprobs",
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
    with pytest.raises(ValueError, match="eager-only"):
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
        max_stream_steps=0,
        temperature=1.0,
        top_p=1.0,
        prompt_lengths=prompt_lengths,
        prompt_repeats=3,
    )
    assert batch.kind.size(0) == 6
    for prompt_index in range(2):
        members = slice(prompt_index * 3, (prompt_index + 1) * 3)
        expected = prompt_ids[prompt_index].expand(3, -1)
        assert torch.equal(batch.token_ids[members], expected)


def test_split_rollout_groups_rejects_mixed_lengths_within_a_group():
    wrapper = _deterministic_wrapper()
    prompt_ids = torch.randint(1, 32, (2, 5))
    lengths = torch.tensor([5, 4])
    batch = rollout_continuations(
        wrapper, prompt_ids, 2, 4, 1.0, 1e-6, prompt_lengths=lengths
    )
    with pytest.raises(ValueError, match="share one prompt length"):
        split_rollout_groups(batch, 2, lengths)
