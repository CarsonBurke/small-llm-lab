from __future__ import annotations

import torch

import train_gpt as baseline
from fresh_lejepa_train_v1_probe_shared_rms_pope import FreshLeJEPASharedRMSV1PoPE
from postraining.core import generalized_advantage_estimate
from postraining.latent_rollout import (
    PAD_SLOT,
    THOUGHT_SLOT,
    TOKEN_SLOT,
    assemble_stream_latents,
    assign_terminal_rewards,
    continuation_reward,
    emitted_token_rows,
    grounded_transition_mask,
    refresh_old_statistics,
    replay_beliefs,
    replay_head_inputs,
    rollout_continuations,
    trim_stream,
)
from postraining.latent_thought import EMIT, THINK, LatentThoughtModel
from postraining.model_io import _pope_construction
from postraining.train_latent_vapo import (
    evaluate_aime_latent,
    sample_prompt_batch,
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
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    for probe in (backbone.policy_probe, backbone.critic_probe):
        for parameter in probe.parameters():
            parameter.requires_grad_(True)
    return LatentThoughtModel(backbone)


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
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    for probe in (backbone.policy_probe, backbone.critic_probe):
        for parameter in probe.parameters():
            parameter.requires_grad_(True)
    return LatentThoughtModel(backbone)


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


def _rollout(wrapper, batch=2, prompt=5, new_tokens=4, max_thinks=2, seed=7):
    prompt_ids = torch.randint(0, 32, (batch, prompt))
    generator = torch.Generator().manual_seed(seed)
    result = rollout_continuations(
        wrapper, prompt_ids, new_tokens, max_thinks, 1.0, 1.0, generator=generator
    )
    return trim_stream(result)


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


def test_rollout_watchdog_bounds_consecutive_thinks():
    wrapper = _wrapper()
    # Bias the gate hard toward THINK so the watchdog must fire.
    with torch.no_grad():
        wrapper.gate.head.bias.fill_(-5.0)
    batch = _rollout(wrapper, batch=2, prompt=4, new_tokens=3, max_thinks=2)
    kinds = batch.kind
    for row in range(kinds.size(0)):
        run = 0
        for slot in kinds[row, batch.prompt_length:].tolist():
            if slot == THOUGHT_SLOT:
                run += 1
                assert run <= 2
            elif slot == TOKEN_SLOT:
                run = 0
    assert float(batch.forced_mask.sum()) > 0
    # Forced EMITs are excluded from the gate PPO mask by construction.
    gate_mask = batch.action_mask * (1.0 - batch.forced_mask)
    assert float((gate_mask * batch.forced_mask).sum()) == 0.0
    # Even a THINK-saturated gate emits exactly the requested tokens: the
    # worst-case max_stream sizing leaves no truncation path.
    for row in emitted_token_rows(batch):
        assert len(row) == 3


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
        predicted = backbone.prediction_latent(beliefs)
        features = torch.cat((stream_inputs, predicted), dim=-1)
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
    unforced = mask & ~batch.forced_mask.bool()
    torch.testing.assert_close(
        gate_logprobs[unforced], batch.old_gate_logprobs[unforced], rtol=2e-4, atol=2e-4
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


def test_grounded_transitions_cover_prompt_and_emissions_but_not_thoughts():
    wrapper = _wrapper()
    batch = _rollout(wrapper)
    grounded = grounded_transition_mask(batch)
    prompt = batch.prompt_length
    assert torch.all(grounded[:, : prompt - 1] == 1.0)
    for row in range(batch.kind.size(0)):
        for position in range(batch.stream_length - 1):
            expected = float(
                batch.kind[row, position] != PAD_SLOT
                and batch.kind[row, position + 1] == TOKEN_SLOT
            )
            assert float(grounded[row, position]) == expected


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


def test_update_minibatch_trains_heads_but_never_the_policy_trunk():
    wrapper = _wrapper()
    critic = _critic()
    backbone = wrapper.backbone
    with torch.no_grad():
        backbone.policy_probe.output.weight.normal_(std=0.02)
    for parameter in wrapper.new_parameters():
        parameter.requires_grad_(False)
    for module in (wrapper.gate, wrapper.transition):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    batch = _rollout(wrapper, batch=4, prompt=5, new_tokens=3)
    assign_terminal_rewards(batch, torch.rand(4))
    trunk_before = backbone.blocks[0].attn.c_qkv.weight.clone()
    # Fresh trunks zero-init output projections, so at step one the gradient
    # reaches attn.proj (whose input is nonzero) but not yet c_qkv behind it.
    critic_trunk_before = critic.trunk.blocks[0].attn.proj.weight.clone()
    gate_before = wrapper.gate.head.weight.clone()
    log_std_before = wrapper.transition.log_std_head.bias.clone()
    optimizers = {
        "gate": torch.optim.AdamW(wrapper.gate.parameters(), lr=1e-2, weight_decay=0.0),
        "renderer": torch.optim.AdamW(
            backbone.policy_probe.parameters(), lr=1e-3, weight_decay=0.0
        ),
        "critic": torch.optim.AdamW(critic.parameters(), lr=1e-3, weight_decay=0.0),
        "transition": torch.optim.AdamW(
            wrapper.transition.parameters(), lr=1e-2, weight_decay=0.0
        ),
    }
    metrics = update_minibatch(wrapper, critic, batch, optimizers, 1e-3, 0.5)
    assert all(
        torch.isfinite(torch.tensor(value)) for value in metrics.values()
    ), metrics
    torch.testing.assert_close(backbone.blocks[0].attn.c_qkv.weight, trunk_before)
    # The separate critic is fully trainable: value CE must reach its trunk.
    assert not torch.equal(
        critic.trunk.blocks[0].attn.proj.weight, critic_trunk_before
    )
    assert not torch.equal(wrapper.transition.log_std_head.bias, log_std_before)
    assert not torch.equal(wrapper.gate.head.weight, gate_before) or float(
        batch.action_mask.sum()
    ) == float(batch.forced_mask.sum())


def test_refresh_old_statistics_matches_the_update_code_path_exactly():
    wrapper = _wrapper()
    critic = _critic()
    batch = _rollout(wrapper, batch=2, prompt=6, new_tokens=4)
    refresh_old_statistics(wrapper, critic, batch)
    backbone = wrapper.backbone
    with torch.no_grad():
        beliefs, _, features, token_targets = replay_head_inputs(wrapper, batch)
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
    optimizers = {
        "gate": torch.optim.AdamW(wrapper.gate.parameters(), lr=1e-3, weight_decay=0.0),
        "renderer": torch.optim.AdamW(
            wrapper.backbone.policy_probe.parameters(), lr=1e-4, weight_decay=0.0
        ),
        "critic": torch.optim.AdamW(critic.parameters(), lr=1e-4, weight_decay=0.0),
        "transition": torch.optim.AdamW(
            wrapper.transition.parameters(), lr=1e-3, weight_decay=0.0
        ),
    }
    metrics = update_minibatch(wrapper, critic, batch, optimizers, 1e-3, 0.5)
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
        max_consecutive_thinks=2, chunk=3, seed=5, device=torch.device("cpu"),
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
        max_consecutive_thinks=2, chunk=2, seed=5, device=torch.device("cpu"),
    )
    assert torch.equal(before, torch.get_rng_state())


def test_positive_lm_loss_applies_only_above_the_reward_threshold():
    wrapper = _wrapper()
    critic = _critic()
    backbone = wrapper.backbone
    batch = _rollout(wrapper, batch=4, prompt=5, new_tokens=3)
    optimizers = {
        "gate": torch.optim.AdamW(wrapper.gate.parameters(), lr=1e-3, weight_decay=0.0),
        "renderer": torch.optim.AdamW(
            backbone.policy_probe.parameters(), lr=1e-4, weight_decay=0.0
        ),
        "critic": torch.optim.AdamW(critic.parameters(), lr=1e-4, weight_decay=0.0),
        "transition": torch.optim.AdamW(
            wrapper.transition.parameters(), lr=1e-3, weight_decay=0.0
        ),
    }
    assign_terminal_rewards(batch, torch.tensor([0.9, 0.1, 0.6, 0.2]))
    metrics = update_minibatch(
        wrapper, critic, batch, optimizers, 1e-3, 0.5, positive_lm_weight=0.1
    )
    assert metrics["positive_fraction"] == 0.5
    # NLL of real emitted tokens under a softcapped 32-way softmax is
    # strictly positive whenever any trajectory qualifies.
    assert metrics["positive_lm_loss"] > 0.0
    # No qualifying trajectory: the loss term is exactly zero.
    assign_terminal_rewards(batch, torch.tensor([0.1, 0.2, 0.3, 0.4]))
    metrics = update_minibatch(
        wrapper, critic, batch, optimizers, 1e-3, 0.5, positive_lm_weight=0.1
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
    update_minibatch(wrapper, critic, batch, optimizers, 0.0, 0.5, value_only=True)
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
