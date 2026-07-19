from __future__ import annotations

import pytest
import torch

import train_gpt as baseline
from fresh_lejepa_train_v1_probe_shared_rms_pope import FreshLeJEPASharedRMSV1PoPE
from postraining.core import (
    generalized_advantage_estimate,
    per_dim_clipped_policy_loss,
)
from postraining.latent_rollout import (
    PAD_SLOT,
    THOUGHT_SLOT,
    TOKEN_SLOT,
    assemble_stream_latents,
    assign_terminal_rewards,
    continuation_reward,
    emitted_token_rows,
    refresh_old_statistics,
    replay_beliefs,
    replay_head_inputs,
    select_thought_actions,
    rollout_continuations,
    split_rollout_groups,
    trim_stream,
)
from postraining.latent_thought import EMIT, THINK, LatentThoughtModel
from postraining.model_io import _pope_construction
from postraining.train_latent_vapo import (
    MathPromptSampler,
    build_optimizers,
    evaluate_aime_latent,
    sample_prompt_batch,
    score_math_rollout,
    think_run_lengths,
    update_minibatch,
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


def _optimizers(
    wrapper, critic, actor_lr=1e-3, head_lr=1e-2, critic_lr=1e-3, renderer_lr=1e-3
):
    """The trainer's actual optimizer layout, at test-scale rates (CPU: unfused)."""
    return build_optimizers(
        wrapper, critic, actor_lr=actor_lr, head_lr=head_lr,
        renderer_lr=renderer_lr, critic_lr=critic_lr, fused=False,
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


def test_emit_only_rollout_still_runs_the_thought_projector_densely():
    wrapper = _wrapper()
    with torch.no_grad():
        wrapper.gate.head.weight.zero_()
        wrapper.gate.head.bias.fill_(100.0)
    projected_shapes = []
    handle = wrapper.backbone.prediction_projector.register_forward_pre_hook(
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


def test_stream_storage_is_internally_consistent():
    wrapper = _wrapper()
    batch = _rollout(wrapper)
    prompt = batch.prompt_length
    assert torch.all(batch.kind[:, :prompt] == TOKEN_SLOT)
    # Every recorded action produced a next input slot of a matching kind.
    action_positions = batch.action_mask.nonzero()
    for row, position in action_positions.tolist():
        action = int(batch.gate_actions[row, position])
        next_kind = int(batch.kind[row, position + 1])
        assert next_kind == (TOKEN_SLOT if action == EMIT else THOUGHT_SLOT)
    # PAD slots carry no thoughts, no actions, no rewards.
    pads = batch.kind == PAD_SLOT
    assert float(batch.action_mask[pads].sum()) == 0.0
    assert float(batch.thoughts[pads].abs().sum()) == 0.0


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
    mask = batch.action_mask.bool()
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
        wrapper, critic, batch, _optimizers(wrapper, critic), gate_pg_coef=0.0
    )
    assert torch.equal(wrapper.gate.head.weight, gate_weight)
    assert torch.equal(wrapper.gate.head.bias, gate_bias)
    # Freezing the gate must not freeze the actor: the trunk still trains
    # through the renderer/thought terms.
    assert not torch.equal(wrapper.backbone.blocks[0].attn.proj.weight, trunk_before)
    assert torch.isfinite(torch.tensor(metrics["gate_loss"]))


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
    batch = _rollout(wrapper, batch=2, prompt=6, new_tokens=4)
    refresh_old_statistics(wrapper, critic, batch)
    backbone = wrapper.backbone
    with torch.no_grad():
        beliefs, predicted, features, token_targets = replay_head_inputs(wrapper, batch)
        values = critic.values(batch).float()
        gate_logprobs = wrapper.gate.log_prob(batch.gate_actions.float(), beliefs)
        token_logprobs = (
            backbone.logits_from_features(features)
            .float()
            .log_softmax(-1)
            .gather(-1, token_targets[..., None])
            .squeeze(-1)
        )
    assert torch.equal(batch.old_values, values)
    assert torch.equal(batch.old_gate_logprobs, gate_logprobs.float())
    assert torch.equal(batch.old_token_logprobs, token_logprobs)
    with torch.no_grad():
        thought_means, thought_targets, think_mask = select_thought_actions(
            batch, predicted
        )
        compact_logprobs = wrapper.transition.per_dim_log_prob(
            thought_targets, thought_means
        )
        thought_logprobs = torch.zeros_like(batch.old_thought_logprobs)
        thought_logprobs[think_mask] = compact_logprobs
    # Epoch-0 per-dim thought ratios are exactly one by construction.
    assert torch.equal(batch.old_thought_logprobs, thought_logprobs.float())


def test_thought_pg_gradient_reaches_the_trunk_through_the_prediction_path():
    # v2 (full-model RL): the thought surrogate alone must backprop through
    # projected thought mean into the trunk — this is the term that retrains the world
    # model from "what WILL come next" toward "what SHOULD come next".
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

    _, predicted, _, _ = replay_head_inputs(wrapper, batch)
    thought_means, thought_targets, think_mask = select_thought_actions(
        batch, predicted
    )
    new_logprobs = wrapper.transition.per_dim_log_prob(
        thought_targets, thought_means
    )
    advantages, _ = generalized_advantage_estimate(
        batch.rewards, batch.old_values, batch.action_mask, torch.ones(4)
    )
    loss, _ = per_dim_clipped_policy_loss(
        new_logprobs,
        batch.old_thought_logprobs[think_mask],
        advantages.detach()[think_mask],
        torch.ones_like(advantages[think_mask]),
    )
    loss.backward()
    proj_grad = wrapper.backbone.blocks[0].attn.proj.weight.grad
    assert proj_grad is not None
    assert float(proj_grad.abs().sum()) > 0.0
    projector_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in wrapper.backbone.prediction_projector.parameters()
        if parameter.grad is not None
    )
    assert projector_grad > 0.0


@pytest.mark.parametrize(
    ("gate_bias", "thought_pg_coef"),
    ((100.0, 1.0), (-2.0, 0.0)),
)
def test_absent_thought_objective_leaves_projector_grad_none_despite_momentum(
    gate_bias: float, thought_pg_coef: float
):
    wrapper = _wrapper()
    critic = _critic()
    optimizers = _optimizers(wrapper, critic)
    projector_parameters = list(wrapper.backbone.prediction_projector.parameters())

    # Seed Adam state so a spurious zero gradient would move the projector
    # through stale momentum even with weight decay disabled.
    optimizers["actor"].zero_grad(set_to_none=True)
    projector_parameters[0].grad = torch.ones_like(projector_parameters[0])
    optimizers["actor"].step()
    optimizers["actor"].zero_grad(set_to_none=True)
    before = [parameter.detach().clone() for parameter in projector_parameters]

    with torch.no_grad():
        wrapper.gate.head.weight.zero_()
        wrapper.gate.head.bias.fill_(gate_bias)
    batch = _rollout(wrapper, batch=4, prompt=5, new_tokens=3)
    think_mask = (batch.gate_actions == THINK) & batch.action_mask.bool()
    if thought_pg_coef:
        assert not bool(think_mask.any())
    else:
        assert bool(think_mask.any())
    assign_terminal_rewards(batch, torch.rand(4))
    refresh_old_statistics(wrapper, critic, batch)
    update_minibatch(
        wrapper,
        critic,
        batch,
        optimizers,
        thought_pg_coef=thought_pg_coef,
    )

    assert all(parameter.grad is None for parameter in projector_parameters)
    for parameter, reference in zip(projector_parameters, before, strict=True):
        torch.testing.assert_close(parameter, reference)


def test_renderer_reads_belief_without_training_the_thought_projector():
    wrapper = _wrapper()
    backbone = wrapper.backbone
    with torch.no_grad():
        backbone.policy_probe.output.weight.normal_(std=0.02)
    batch = _rollout(wrapper, batch=2, prompt=5, new_tokens=3)
    stream_inputs, beliefs = replay_beliefs(wrapper, batch)
    predicted = backbone.prediction_latent(beliefs)
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
        parameter.grad is None for parameter in backbone.prediction_projector.parameters()
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
    # Three groups: trunk, fresh heads, renderer probe at its own rate.
    lrs = [group["lr"] for group in optimizers["actor"].param_groups]
    assert lrs == [1e-3, 1e-2, 1e-3]
    renderer_group = optimizers["actor"].param_groups[2]["params"]
    assert {id(p) for p in renderer_group} == {
        id(p) for p in wrapper.backbone.policy_probe.parameters()
    }


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
        wrapper, critic, batch, optimizers, actor_step=False, loss_scale=0.5
    )
    # The actor did not step, but its gradient is banked for the caller.
    torch.testing.assert_close(trunk_weight.detach(), before)
    assert metrics["trunk_grad_norm"] > 0.0
    assert trunk_weight.grad is not None
    optimizers["actor"].step()
    assert not torch.equal(trunk_weight.detach(), before)


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
        _optimizers(wrapper, critic, actor_lr=1e-4, head_lr=1e-3, critic_lr=1e-4),
    )
    assert all(
        torch.isfinite(torch.tensor(value)) for value in metrics.values()
    ), metrics
    # With refreshed old statistics, epoch-0 ratios started at exactly one,
    # so nothing clipped on the first update.
    assert metrics["gate_clip_fraction"] == 0.0
    assert metrics["renderer_clip_fraction"] == 0.0


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


def test_evaluate_aime_latent_scores_through_the_gate_policy():
    wrapper = _wrapper()

    class _Tokenizer:
        def eos_id(self) -> int:
            return -1

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
    metrics = evaluate_aime_latent(
        wrapper, _Tokenizer(), rows, samples=4, max_new_tokens=3,
        max_stream_steps=12, chunk=3, seed=5, device=torch.device("cpu"),
        prompt_tokens=8,
    )
    # Every decode reads "Answer: 42": row one is always right, row two
    # always wrong, so accuracy pins both counting and verification.
    assert metrics["samples"] == 8
    assert metrics["accuracy"] == 0.5
    assert 0.0 <= metrics["think_fraction"] <= 1.0
    # The eval must not perturb training RNG state.
    before = torch.get_rng_state()
    evaluate_aime_latent(
        wrapper, _Tokenizer(), rows[:1], samples=2, max_new_tokens=2,
        max_stream_steps=8, chunk=2, seed=5, device=torch.device("cpu"),
        prompt_tokens=8,
    )
    assert torch.equal(before, torch.get_rng_state())


def test_positive_lm_loss_applies_only_above_the_reward_threshold():
    wrapper = _wrapper()
    critic = _critic()
    backbone = wrapper.backbone
    batch = _rollout(wrapper, batch=4, prompt=5, new_tokens=3)
    optimizers = _optimizers(
        wrapper, critic, actor_lr=1e-4, head_lr=1e-3, critic_lr=1e-4
    )
    assign_terminal_rewards(batch, torch.tensor([0.9, 0.1, 0.6, 0.2]))
    metrics = update_minibatch(
        wrapper, critic, batch, optimizers, positive_lm_weight=0.1
    )
    assert metrics["positive_fraction"] == 0.5
    # NLL of real emitted tokens under a softcapped 32-way softmax is
    # strictly positive whenever any trajectory qualifies.
    assert metrics["positive_lm_loss"] > 0.0
    # No qualifying trajectory: the loss term is exactly zero.
    assign_terminal_rewards(batch, torch.tensor([0.1, 0.2, 0.3, 0.4]))
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


def test_math_prompt_sampler_resumes_deterministically_across_epochs():
    rows = [{"id": index} for index in range(7)]
    full = MathPromptSampler(rows, seed=3).next_rows(10)
    # Ten draws from seven rows crosses an epoch boundary; the first epoch
    # must be a permutation of the corpus, not a with-replacement sample.
    assert sorted(row["id"] for row in full[:7]) == list(range(7))
    resumed = MathPromptSampler(rows, seed=3)
    resumed.cursor = 4
    assert resumed.next_rows(6) == full[4:]
    # A different seed reorders the stream.
    assert MathPromptSampler(rows, seed=4).next_rows(10) != full


def test_score_math_rollout_scores_the_verifier_on_truncated_decodes():
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
    score_math_rollout(batch, "42", tokenizer, (eos,))
    assert batch.reward_scalar.tolist() == [1.0, 1.0, 1.0, 1.0]
    # One decode per row, each EOS-truncated: EOS may only appear last.
    assert len(tokenizer.calls) == 4
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
    assert batch.reward_scalar.tolist() == [0.0, 0.0, 0.0, 0.0]


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

    import postraining.train_latent_vapo as trainer

    monkeypatch.setattr(trainer, "rollout_continuations", _spy)
    # Both boundary tokens must reach the rollout so eval rows stop exactly
    # like training rollouts; -1 sentinels must drop out, and no tokens at
    # all must map to None (no stop handling).
    for eos, bos, expected in ((5, 4, (5, 4)), (5, -1, (5,)), (-1, -1, None)):
        evaluate_aime_latent(
            wrapper, _Tokenizer(eos, bos), rows, samples=2, max_new_tokens=2,
            max_stream_steps=8, chunk=2, seed=5, device=torch.device("cpu"),
            prompt_tokens=8,
        )
        assert seen and seen[-1] == expected


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

    import postraining.train_latent_vapo as trainer

    monkeypatch.setattr(trainer, "rollout_continuations", _spy)
    # The question and answer-format instruction sit at the END of DAPO/AIME
    # prompts, so truncation must keep the tail — behind the BOS that frames
    # the prompt as a document start, matching the training path.
    evaluate_aime_latent(
        wrapper, _Tokenizer(), rows, samples=2, max_new_tokens=2,
        max_stream_steps=8, chunk=2, seed=5, device=torch.device("cpu"),
        prompt_tokens=5,
    )
    assert seen and seen[-1].shape == (2, 5)
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
        wrapper, prompt_ids, 4, 8, 1.0, 1e-6, prompt_lengths=prompt_lengths
    )
    groups = split_rollout_groups(batched, samples, prompt_lengths)

    for group, prompt in zip(groups, prompts, strict=True):
        torch.manual_seed(29)
        expected = rollout_continuations(
            wrapper, prompt[None].expand(samples, -1), 4, 8, 1.0, 1e-6
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
        assert torch.equal(group.emit_mask, expected.emit_mask)
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


def test_batched_rollout_rejects_bad_prompt_lengths_and_static_caches():
    wrapper = _wrapper()
    prompt_ids = torch.randint(1, 32, (2, 5))
    with pytest.raises(ValueError, match="one true length per row"):
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


def test_split_rollout_groups_rejects_mixed_lengths_within_a_group():
    wrapper = _deterministic_wrapper()
    prompt_ids = torch.randint(1, 32, (2, 5))
    lengths = torch.tensor([5, 4])
    batch = rollout_continuations(
        wrapper, prompt_ids, 2, 4, 1.0, 1e-6, prompt_lengths=lengths
    )
    with pytest.raises(ValueError, match="share one prompt length"):
        split_rollout_groups(batch, 2, lengths)
