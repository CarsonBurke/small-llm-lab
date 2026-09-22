"""Deterministic hidden-carry (``--reasoning-mode carry``) invariants.

Carry is a token-only policy whose generated-token inputs add, through a
learned zero-init combiner, the detached belief that produced that token.
These tests pin the contract on the nano (RoPE) and KDA (recurrent) trunks:
bitwise-cot at initialization, exact rollout/replay parity with a live
combiner, causality of the stored carry for actor and critic, row alignment
through left padding, packing and arena fillers, the refusals that keep a
carry stream from being read as any other policy, and the schema bytes the
in-flight token-only and latent runs resume under.
"""

from __future__ import annotations

import copy
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from postraining.latent_rollout import (
    PAD_SLOT,
    TOKEN_SLOT,
    assign_terminal_rewards,
    generated_slot_mask,
    pack_rollout_groups_for_replay,
    refresh_old_statistics,
    replay_beliefs,
    rollout_continuations,
    split_rollout_groups,
    trim_stream,
    validate_hidden_carry_rollout,
)
from postraining.latent_thought import (
    HIDDEN_CARRY_INPUT_SCHEMA,
    HIDDEN_CARRY_ROLLOUT_POLICY_SCHEMA,
    PINNED_EMIT_ROLLOUT_POLICY_SCHEMAS,
    ROLLOUT_POLICY_SCHEMA,
    THOUGHT_DISTRIBUTION_SCHEMA,
    THOUGHT_INPUT_SCHEMA,
    DecodeRangeMask,
    LatentThoughtModel,
    combiner_init_kwargs_from_checkpoint,
    rollout_policy_schema_for_mode,
    validate_renderer_checkpoint,
)
from postraining.nano_backbone import NanoGPTBackbone
from postraining.reasoning_modes import (
    REASONING_MODES,
    mode_carries_hidden,
    mode_pins_emit,
    mode_rollout_budget,
)
from postraining.rollout_scheduler import rollout_continuous_refill_groups
from postraining.tests.test_kda_backbone import MODEL_KWARGS as KDA_KWARGS
from postraining.tests.test_kda_backbone import _seeded_backbone as _kda_backbone
from postraining.train_latent_vapo import (
    MathPromptSampler,
    aggregate_actor_tensorboard_metrics,
    build_optimizers,
    save_checkpoint,
    update_minibatch,
)
from postraining.value_model import SeparateCritic
from postraining.vapo import schemas
from postraining.vapo.config import build_arg_parser, validate_args

NANO_KWARGS = dict(vocab_size=64, num_layers=2, model_dim=256)


def _nano_backbone(seed: int = 3) -> NanoGPTBackbone:
    torch.manual_seed(seed)
    backbone = NanoGPTBackbone(**NANO_KWARGS).float().eval()
    with torch.no_grad():
        # A zero-init readout makes every logit comparison 0 == 0.
        backbone.proj.weight.normal_(std=0.05)
        backbone.proj.bias.normal_(std=0.05)
    return backbone


BACKBONES = {
    "nano": (_nano_backbone, NANO_KWARGS["vocab_size"]),
    "kda": (_kda_backbone, KDA_KWARGS["vocab_size"]),
}
# Stepwise decode against full-sequence replay: the recurrent trunk's step
# and chunked reference recurrences reduce in different orders.
PARITY_TOLERANCE = {"nano": 1e-4, "kda": 2e-4}


def _live_combiner(combiner, seed: int) -> None:
    """Give every combiner path real weights, so the carry is load-bearing."""
    generator = torch.Generator().manual_seed(seed)
    dim = combiner.carry.weight.size(0)

    def draw(parameter, std):
        parameter.copy_(
            torch.randn(parameter.shape, generator=generator) * std
        )

    with torch.no_grad():
        draw(combiner.carry.weight, 0.5 / dim**0.5)
        draw(combiner.type_bias, 0.05)
        for mlp in combiner.mlps:
            draw(mlp.proj.weight, 0.02)


def _carry_wrapper(kind: str = "nano", *, live: bool = True, seed: int = 3):
    factory, _ = BACKBONES[kind]
    wrapper = LatentThoughtModel(factory(seed), hidden_carry=True).eval()
    if live:
        _live_combiner(wrapper.combiner, seed + 100)
    return wrapper


def _critic(kind: str = "nano", *, live: bool = False, seed: int = 11):
    factory, _ = BACKBONES[kind]
    critic = SeparateCritic(factory(seed)).eval()
    with torch.no_grad():
        # The zero-weight prior head makes every value the constant prior.
        critic.head.weight.normal_(std=0.05)
    if live:
        _live_combiner(critic.combiner, seed + 100)
    return critic


def _prompts(kind: str, rows: int, length: int, seed: int = 5):
    _, vocab = BACKBONES[kind]
    return torch.randint(
        1, vocab, (rows, length), generator=torch.Generator().manual_seed(seed)
    )


def _roll(
    wrapper,
    prompt_ids,
    *,
    new_tokens: int = 6,
    seed: int = 7,
    hidden_carry: bool = True,
    trim: bool = True,
    top_p: float = 1.0,
    **kwargs,
):
    """A pinned-EMIT rollout; carry on by default, stopping on a quarter of ids.

    ``top_p`` near zero is the argmax path, for comparisons whose rows draw
    from differently shaped batches.
    """
    generator = torch.Generator().manual_seed(seed)
    kwargs.setdefault("stop_ids", tuple(range(1, 17)))
    with torch.no_grad():
        batch = rollout_continuations(
            wrapper,
            prompt_ids,
            new_tokens,
            new_tokens,
            1.0,
            top_p,
            generator=generator,
            pin_emit=True,
            hidden_carry=hidden_carry,
            **kwargs,
        )
    return trim_stream(batch) if trim else batch


def _replayed_token_logprobs(wrapper, batch):
    stream_inputs, beliefs = replay_beliefs(wrapper, batch)
    features = wrapper.renderer_features(stream_inputs, beliefs)
    logits = wrapper.backbone.logits_from_features(features).float()
    targets = torch.zeros_like(batch.token_ids)
    targets[:, :-1] = batch.token_ids[:, 1:]
    logprobs = logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
    return logprobs, beliefs


def _zeroed_hiddens(batch):
    zeroed = copy.deepcopy(batch)
    zeroed.hiddens.zero_()
    return zeroed


def _cot_args(**overrides):
    return SimpleNamespace(value_warmup_steps=0, answer_fence=True, **overrides)


# --- mode flags and schema bytes --------------------------------------------


def test_reasoning_mode_flags():
    assert REASONING_MODES == ("latent", "carry", "cot", "none")
    assert [mode_pins_emit(mode) for mode in REASONING_MODES] == [
        False, True, True, True,
    ]
    assert [mode_carries_hidden(mode) for mode in REASONING_MODES] == [
        False, True, False, False,
    ]
    # Carry spends its whole stream budget on tokens, exactly like cot.
    assert mode_rollout_budget("carry", 700, answer_tokens=8) == (700, 700)


def test_token_only_and_latent_schema_bytes_are_pinned():
    """In-flight cot/none/latent runs resume against these exact bytes."""
    v29 = (
        "unique_prefix_compact_tail_broad_mixture_forced_initial_think_"
        "one_way_stop_vector_sigma_isotropic_trajectory_position_rng/v29"
    )
    refill = v29 + "+request_stable_gate_token_gaussian_refill_paged_flex/v3"
    replay = "compact_emit_gate_raw_gaussian_next_slot_exact_replay/v4"
    objectives = (
        "vapo_joint_gate_token_gaussian_clip_token_denominator/v4",
        "delightful_token_policy_plus_vapo_gate_gaussian/v2",
        "target_policy_token_odds_plus_vapo_gate_gaussian/v7",
    )
    for mode in ("latent", "cot", "none"):
        assert schemas.execution_schema_for(mode) == v29
        assert schemas.execution_schema_for(mode, "continuous_refill") == refill
        assert schemas.replay_numerics_schema_for(mode) == replay
        assert (
            schemas.actor_objective_schema(mode),
            schemas.actor_objective_schema(mode, True),
            schemas.actor_objective_schema(mode, False, True),
        ) == objectives
    assert rollout_policy_schema_for_mode("latent") == (
        "forced_initial_think_one_way_stop_isotropic/v3"
    )
    assert PINNED_EMIT_ROLLOUT_POLICY_SCHEMAS == {
        "cot": "pinned_emit_token_only_cot/v2",
        "none": "pinned_emit_token_only_answer_prefix/v2",
    }
    assert THOUGHT_INPUT_SCHEMA == "combined_raw_gaussian_prenorm_mlp/v3"
    assert THOUGHT_DISTRIBUTION_SCHEMA == (
        "belief_centered_isotropic_vector_sigma_over_sqrt_dim/v2"
    )


def test_carry_schemas_are_distinct_and_lockstep_only():
    assert schemas.execution_schema_for("carry") == (
        "unique_prefix_compact_tail_broad_mixture_deterministic_hidden_carry_"
        "token_only_generator_token_rng/v30"
    )
    assert schemas.replay_numerics_schema_for("carry") == (
        "compact_token_logprob_next_slot_hidden_carry_exact_replay/v5"
    )
    assert (
        schemas.actor_objective_schema("carry"),
        schemas.actor_objective_schema("carry", True),
        schemas.actor_objective_schema("carry", False, True),
    ) == (
        "vapo_token_clip_hidden_carry_token_denominator/v4",
        "delightful_token_policy_hidden_carry/v2",
        "target_policy_token_odds_hidden_carry/v7",
    )
    assert rollout_policy_schema_for_mode("carry") == (
        "deterministic_hidden_carry/v2"
    )
    assert HIDDEN_CARRY_INPUT_SCHEMA == "zero_init_hidden_residual_prenorm_mlp/v2"
    carry = {
        schemas.execution_schema_for("carry"),
        schemas.replay_numerics_schema_for("carry"),
        schemas.actor_objective_schema("carry"),
    }
    for mode in ("latent", "cot", "none"):
        assert not carry & {
            schemas.execution_schema_for(mode),
            schemas.replay_numerics_schema_for(mode),
            schemas.actor_objective_schema(mode),
        }
    with pytest.raises(ValueError, match="lockstep scheduler only"):
        schemas.execution_schema_for("carry", "continuous_refill")
    for function in (
        schemas.execution_schema_for,
        schemas.replay_numerics_schema_for,
        schemas.actor_objective_schema,
    ):
        with pytest.raises(ValueError, match="unknown reasoning mode"):
            function("thought")
    # No migrations in either direction.
    latent_payload = {
        "execution_schema": schemas.execution_schema_for("latent"),
        "replay_numerics_schema": schemas.replay_numerics_schema_for("latent"),
    }
    assert not schemas.resume_execution_schema_compatible(
        latent_payload,
        expected_execution_schema=schemas.execution_schema_for("carry"),
    )
    assert not schemas.resume_replay_schema_compatible(latent_payload, "carry")


# --- rollout ------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(BACKBONES))
def test_zero_init_carry_rollout_is_bitwise_the_cot_rollout(kind):
    factory, _ = BACKBONES[kind]
    backbone = factory(3)
    cot = LatentThoughtModel(backbone).eval()
    carry = LatentThoughtModel(backbone, hidden_carry=True).eval()
    prompt_ids = _prompts(kind, 6, 5)
    expected = _roll(cot, prompt_ids, hidden_carry=False)
    actual = _roll(carry, prompt_ids)

    assert actual.carry_injected and not expected.carry_injected
    assert expected.hiddens.size(-1) == 0
    for field in fields(expected):
        if field.name in ("hiddens", "carry_injected", "replay_layout_token"):
            continue
        left, right = getattr(expected, field.name), getattr(actual, field.name)
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right), field.name
        else:
            assert left == right, field.name
    # The rollout must actually be ragged for the length contract to bite.
    actions = actual.action_mask.sum(1)
    assert len(set(actions.tolist())) > 1
    # Stream length is prompt + emitted tokens: no thought or gate slot.
    real = (actual.kind != PAD_SLOT).sum(1)
    assert torch.equal(real, actual.prompt_length + actions.long())
    assert bool((actual.kind[actual.kind != PAD_SLOT] == TOKEN_SLOT).all())
    validate_hidden_carry_rollout(actual)
    # A stored belief at every generated slot, and nothing anywhere else.
    generated = generated_slot_mask(actual)
    assert bool((actual.hiddens[generated].abs().sum(-1) > 0).all())
    assert float(actual.hiddens[~generated].abs().sum()) == 0.0
    assert actual.hiddens.dtype == torch.float32


@pytest.mark.parametrize("kind", sorted(BACKBONES))
def test_live_carry_rollout_replays_exactly(kind):
    wrapper = _carry_wrapper(kind)
    batch = _roll(wrapper, _prompts(kind, 6, 5))
    tolerance = PARITY_TOLERANCE[kind]
    with torch.no_grad():
        replayed, beliefs = _replayed_token_logprobs(wrapper, batch)
        zeroed, _ = _replayed_token_logprobs(wrapper, _zeroed_hiddens(batch))
    emits = batch.emit_mask.bool()
    assert bool(emits.any())
    torch.testing.assert_close(
        replayed[emits],
        batch.old_token_logprobs[emits],
        rtol=tolerance,
        atol=tolerance,
    )
    # hiddens[p] is the belief replay recomputes at p - 1: the carry records
    # exactly the state that sampled the token it rides on.
    carried = generated_slot_mask(batch)[:, 1:]
    torch.testing.assert_close(
        batch.hiddens[:, 1:][carried],
        beliefs[:, :-1][carried],
        rtol=tolerance,
        atol=tolerance,
    )
    # And the live combiner makes that record load-bearing.
    assert (
        (zeroed[emits] - batch.old_token_logprobs[emits]).abs().max()
        > 100 * tolerance
    )


@pytest.mark.parametrize("kind", sorted(BACKBONES))
def test_a_carried_hidden_only_affects_its_own_row_from_its_slot_on(kind):
    wrapper = _carry_wrapper(kind)
    critic = _critic(kind, live=True)
    batch = _roll(wrapper, _prompts(kind, 4, 5), stop_ids=())
    generated = generated_slot_mask(batch)
    slots = generated[0].nonzero().squeeze(-1)
    slot = int(slots[len(slots) // 2])
    perturbed = copy.deepcopy(batch)
    perturbed.hiddens[0, slot] += torch.randn(
        batch.hiddens.size(-1), generator=torch.Generator().manual_seed(1)
    )
    with torch.no_grad():
        _, before = replay_beliefs(wrapper, batch)
        _, after = replay_beliefs(wrapper, perturbed)
        values_before = critic.values(batch)
        values_after = critic.values(perturbed)
    for name, left, right in (
        ("actor", before, after),
        ("critic", values_before, values_after),
    ):
        torch.testing.assert_close(
            right[0, :slot], left[0, :slot], rtol=0, atol=1e-6, msg=name
        )
        torch.testing.assert_close(right[1:], left[1:], rtol=0, atol=1e-6, msg=name)
        assert (right[0, slot] - left[0, slot]).abs().max() > 1e-3, name


def test_carry_alignment_survives_left_padding_split_and_packing():
    wrapper = _carry_wrapper()
    prompts = [_prompts("nano", 1, length, seed)[0] for length, seed in (
        (7, 11), (4, 12), (6, 13),
    )]
    samples = 2
    width = max(prompt.numel() for prompt in prompts)
    prompt_ids = torch.zeros((len(prompts) * samples, width), dtype=torch.long)
    for index, prompt in enumerate(prompts):
        rows = slice(index * samples, (index + 1) * samples)
        prompt_ids[rows, width - prompt.numel():] = prompt
    prompt_lengths = torch.tensor(
        [prompt.numel() for prompt in prompts]
    ).repeat_interleave(samples)
    batched = _roll(
        wrapper,
        prompt_ids,
        top_p=1e-6,
        prompt_lengths=prompt_lengths,
        trim=False,
    )
    groups = [
        trim_stream(group)
        for group in split_rollout_groups(batched, samples, prompt_lengths)
    ]
    for group, prompt in zip(groups, prompts, strict=True):
        expected = _roll(wrapper, prompt[None].expand(samples, -1), top_p=1e-6)
        assert torch.equal(group.token_ids, expected.token_ids)
        assert torch.equal(group.action_mask, expected.action_mask)
        torch.testing.assert_close(
            group.hiddens, expected.hiddens, rtol=1e-4, atol=1e-5
        )
        prompt_length = group.prompt_length
        # The last prompt slot is a plain read token; the first generated
        # slot carries the prompt-last belief.
        assert float(group.hiddens[:, : prompt_length].abs().sum()) == 0.0
        with torch.no_grad():
            _, beliefs = replay_beliefs(wrapper, group)
        torch.testing.assert_close(
            group.hiddens[:, prompt_length],
            beliefs[:, prompt_length - 1],
            rtol=1e-4,
            atol=1e-4,
        )
        # Nothing is stored past a row's final token.
        last = (group.kind != PAD_SLOT).sum(1)
        for row, end in enumerate(last.tolist()):
            assert float(group.hiddens[row, end:].abs().sum()) == 0.0

    packed = pack_rollout_groups_for_replay(groups)
    assert packed.carry_injected
    critic = _critic(live=True)
    with torch.no_grad():
        _, packed_beliefs = replay_beliefs(wrapper, packed)
        packed_values = critic.values(packed)
    row_start = 0
    for group in groups:
        rows = slice(row_start, row_start + samples)
        with torch.no_grad():
            _, beliefs = replay_beliefs(wrapper, group)
            values = critic.values(group)
        # Per-row generated-slot flags survive the packed batch's scalar
        # prompt length, which is only the minimum across groups.
        assert torch.equal(
            generated_slot_mask(packed)[rows, : group.stream_length],
            generated_slot_mask(group),
        )
        torch.testing.assert_close(
            packed_beliefs[rows, : group.stream_length],
            beliefs,
            rtol=2e-4,
            atol=2e-5,
        )
        torch.testing.assert_close(
            packed_values[rows, : group.stream_length],
            values,
            rtol=2e-4,
            atol=2e-5,
        )
        row_start += samples


# Argmax stop sets chosen so this trunk's greedy rows end at different steps.
STOP_IDS = {"nano": (50, 57), "kda": (15, 24)}


@pytest.mark.parametrize("kind", sorted(BACKBONES))
def test_carry_survives_fan_out_compaction_and_the_static_tail(kind):
    """Finished-row compaction and the static tail move rows, not carries.

    ``sync_every=1`` compacts after every step, and ``finished_batch_size``
    lands the survivors in caller-owned tail caches, so each carry is
    written through a remapped row index. The argmax path makes the
    uncompacted rollout the exact reference.
    """
    wrapper = _carry_wrapper(kind)
    prompt_ids = _prompts(kind, 4, 5, seed=19)
    common = dict(
        new_tokens=12,
        top_p=1e-6,
        prompt_repeats=2,
        tensor_positions=True,
        stop_ids=STOP_IDS[kind],
    )
    reference = _roll(wrapper, prompt_ids, compact_finished=False, **common)
    compacted = _roll(
        wrapper,
        prompt_ids,
        compact_finished=True,
        sync_every=1,
        finished_batch_size=4,
        tail_caches=wrapper.make_static_generation_cache(
            4, 5 + 12, torch.device("cpu")
        ),
        **common,
    )
    actions = reference.action_mask.sum(1)
    # Ragged, so rows finish and compact while others keep decoding.
    assert len(set(actions.tolist())) > 1 and int(actions.max()) >= 6
    for field in fields(reference):
        if field.name == "replay_layout_token":
            continue
        left, right = getattr(reference, field.name), getattr(compacted, field.name)
        if isinstance(left, torch.Tensor):
            # Static tail caches attend a masked full-width window, which
            # reorders fp32 reductions at the 1e-6 level on the KDA trunk.
            torch.testing.assert_close(right, left, rtol=0, atol=1e-5, msg=field.name)
        else:
            assert left == right, field.name
    with torch.no_grad():
        _, beliefs = replay_beliefs(wrapper, compacted)
    carried = generated_slot_mask(compacted)[:, 1:]
    torch.testing.assert_close(
        compacted.hiddens[:, 1:][carried],
        beliefs[:, :-1][carried],
        rtol=PARITY_TOLERANCE[kind],
        atol=PARITY_TOLERANCE[kind],
    )


def test_arena_filler_rows_leave_the_carry_record_unchanged():
    """Static-arena fillers must not move a real row's carry or tokens.

    Pinned-EMIT tokens come from one batch-shaped generator draw, so appended
    rows shift a sampled row's draws for cot and carry alike; the argmax path
    isolates the arena's own contribution, which must be none.
    """
    wrapper = _carry_wrapper()
    prompt_ids = _prompts("nano", 4, 4, seed=31)
    prompt_ids[:2, 0] = 0
    prompt_lengths = torch.tensor([3, 3, 4, 4])
    repeats, width = 2, 32
    rows = prompt_ids.size(0) * repeats
    cpu = torch.device("cpu")

    def run(arena_rows):
        return _roll(
            wrapper,
            prompt_ids,
            new_tokens=width - prompt_ids.size(1),
            top_p=1e-6,
            trim=False,
            prompt_lengths=prompt_lengths,
            prompt_repeats=repeats,
            tensor_positions=True,
            compact_finished=False,
            caches=wrapper.make_static_generation_cache(arena_rows, width, cpu),
            decode_mask=DecodeRangeMask(
                arena_rows, width, cpu, block_size=16, row_bucket=8
            ),
        )

    exact = run(rows)
    padded = run(rows + 2 * repeats)
    assert exact.carry_injected and padded.carry_injected
    for field in fields(exact):
        if field.name == "replay_layout_token":
            continue
        left, right = getattr(exact, field.name), getattr(padded, field.name)
        if not isinstance(left, torch.Tensor):
            assert left == right, field.name
            continue
        assert right.size(0) == rows, field.name
        torch.testing.assert_close(right, left, msg=field.name)
    # Without compaction, ended rows keep stepping; they must record nothing.
    generated = generated_slot_mask(exact)
    assert bool(generated.any())
    assert bool((exact.action_mask.sum(1) < width - prompt_ids.size(1)).any())
    assert bool((exact.hiddens[generated].abs().sum(-1) > 0).all())
    assert float(exact.hiddens[~generated].abs().sum()) == 0.0


# --- replay, update, and optimizer ----------------------------------------------


def test_refresh_then_update_is_age_zero_exact_and_trains_only_carry_heads():
    wrapper = _carry_wrapper()
    critic = _critic()
    batch = _roll(wrapper, _prompts("nano", 4, 5))
    assign_terminal_rewards(batch, torch.tensor([1.0, 0.0, 1.0, 0.0]))
    refresh_old_statistics(wrapper, critic, batch)
    # No gate exists, so there is no gate log-probability to record.
    assert float(batch.old_stop_logprobs.abs().sum()) == 0.0
    optimizers = build_optimizers(wrapper, critic, learning_rate=1e-3, fused=False)
    metrics = update_minibatch(
        wrapper, critic, batch, optimizers, actor_step=False, critic_step=False
    )
    assert metrics["token_abs_log_ratio_max"] == 0.0
    assert metrics["policy_clip_fraction"] == 0.0
    assert metrics["carry_input_delta_rms_ratio"] > 0.0
    for absent in (
        "stochastic_policy_loss",
        "stochastic_policy_clip_fraction",
        "gate_grad_norm",
        "thought_mean_grad_norm",
    ):
        assert absent not in metrics
    dashboard = aggregate_actor_tensorboard_metrics([metrics, metrics])
    assert dashboard["carry/input_delta_rms_ratio"] == pytest.approx(
        metrics["carry_input_delta_rms_ratio"]
    )
    for absent in (
        "loss/stochastic_policy",
        "clip/stochastic_policy",
        "grad/gate",
        "grad/thought_mean",
    ):
        assert absent not in dashboard


def test_zero_init_carry_learns_from_the_first_update():
    """The zero-init map has a full-rank first-step gradient (loss outer hidden)."""
    wrapper = _carry_wrapper(live=False)
    critic = _critic()
    batch = _roll(wrapper, _prompts("nano", 4, 5))
    assign_terminal_rewards(batch, torch.tensor([1.0, 0.0, 1.0, 0.0]))
    refresh_old_statistics(wrapper, critic, batch)
    carry_before = wrapper.combiner.carry.weight.clone()
    metrics = update_minibatch(
        wrapper,
        critic,
        batch,
        build_optimizers(wrapper, critic, learning_rate=1e-3, fused=False),
    )
    assert metrics["carry_input_delta_rms_ratio"] == 0.0
    assert not torch.equal(wrapper.combiner.carry.weight, carry_before)


def test_critic_reads_the_carry_through_its_own_combiner():
    wrapper = _carry_wrapper()
    critic = _critic()
    assert critic.combiner is not wrapper.combiner
    batch = _roll(wrapper, _prompts("nano", 4, 5))
    with torch.no_grad():
        # A fresh critic combiner is an identity: the actor's live combiner
        # never leaks into the critic's inputs.
        assert torch.equal(
            critic.values(batch), critic.values(_zeroed_hiddens(batch))
        )
    assign_terminal_rewards(batch, torch.tensor([1.0, 0.0, 1.0, 0.0]))
    refresh_old_statistics(wrapper, critic, batch)
    actor_before = copy.deepcopy(wrapper.state_dict())
    critic_carry_before = critic.combiner.carry.weight.clone()
    update_minibatch(
        wrapper,
        critic,
        batch,
        build_optimizers(wrapper, critic, learning_rate=1e-3, fused=False),
        value_only=True,
    )
    # Value loss trains the critic's carry map and never the actor's.
    assert not torch.equal(critic.combiner.carry.weight, critic_carry_before)
    for name, value in wrapper.state_dict().items():
        assert torch.equal(value, actor_before[name]), name
    with torch.no_grad():
        assert not torch.equal(
            critic.values(batch), critic.values(_zeroed_hiddens(batch))
        )


def test_carry_optimizer_layout_holds_only_the_combiner_as_new_parameters():
    wrapper = _carry_wrapper()
    assert wrapper.gate is None and wrapper.transition is None
    assert wrapper.thought_input_schema == HIDDEN_CARRY_INPUT_SCHEMA
    assert wrapper.thought_distribution_schema is None
    new = list(wrapper.new_parameters())
    assert {id(p) for p in new} == {id(p) for p in wrapper.combiner.parameters()}
    optimizers = build_optimizers(wrapper, _critic(), learning_rate=1e-3, fused=False)
    groups = optimizers["actor"].param_groups
    assert [id(p) for p in groups[1]["params"]] == [id(p) for p in new]
    assert not any(
        key.startswith(("gate.", "transition.")) for key in wrapper.state_dict()
    )


# --- refusals -------------------------------------------------------------------


def test_carry_refuses_stochastic_execution_and_mixed_streams():
    carry = _carry_wrapper()
    prompt_ids = _prompts("nano", 2, 4)
    with pytest.raises(ValueError, match="needs the stop gate"):
        rollout_continuations(carry, prompt_ids, 2, 4, 1.0, 1.0)
    cot = LatentThoughtModel(_nano_backbone()).eval()
    with pytest.raises(ValueError, match="pinned-EMIT token policy"):
        rollout_continuations(carry, prompt_ids, 2, 4, 1.0, 1.0, hidden_carry=True)
    with pytest.raises(ValueError, match="needs a hidden-carry wrapper"):
        rollout_continuations(
            cot, prompt_ids, 2, 4, 1.0, 1.0, pin_emit=True, hidden_carry=True
        )
    with pytest.raises(ValueError):
        LatentThoughtModel(_nano_backbone(), hidden_carry=True, thought_sigma=1.0)
    with pytest.raises(ValueError):
        LatentThoughtModel(
            _nano_backbone(),
            hidden_carry=True,
            init_stop_thinking_probability=0.9,
        )

    carried = _roll(carry, prompt_ids)
    token_only = _roll(cot, prompt_ids, hidden_carry=False)
    with pytest.raises(ValueError, match="cannot pack carry-injected"):
        pack_rollout_groups_for_replay([carried, token_only])
    # Evaluation-shaped rollouts drop the record and cannot be replayed.
    discarded = _roll(carry, prompt_ids, replay_storage=False)
    assert discarded.carry_injected and discarded.hiddens.size(-1) == 0
    with pytest.raises(ValueError, match="discarded its carried hiddens"):
        replay_beliefs(carry, discarded)
    with pytest.raises(ValueError, match="discarded its carried hiddens"):
        _critic().values(discarded)
    # A token-only batch that somehow stored hiddens is refused, not read.
    forged = copy.deepcopy(token_only)
    forged.hiddens = carried.hiddens.clone()
    with pytest.raises(ValueError, match="never injected"):
        replay_beliefs(cot, forged)
    # Neither policy replays the other's streams.
    with pytest.raises(ValueError, match="rollout carry does not match"):
        replay_beliefs(cot, carried)
    with pytest.raises(ValueError, match="rollout carry does not match"):
        replay_beliefs(carry, token_only)
    with pytest.raises(ValueError, match="never injected"):
        _critic().values(forged)
    # The update refuses to optimize one policy on the other's streams.
    critic = _critic()
    optimizers = build_optimizers(cot, critic, learning_rate=1e-3, fused=False)
    with pytest.raises(ValueError, match="rollout carry does not match"):
        update_minibatch(cot, critic, carried, optimizers)
    optimizers = build_optimizers(carry, critic, learning_rate=1e-3, fused=False)
    with pytest.raises(ValueError, match="rollout carry does not match"):
        update_minibatch(carry, critic, token_only, optimizers)


def test_carry_rollout_validation_refuses_non_token_actions():
    batch = _roll(_carry_wrapper(), _prompts("nano", 2, 4))
    validate_hidden_carry_rollout(batch)
    gated = copy.deepcopy(batch)
    gated.stop_mask[0, batch.prompt_length - 1] = 1.0
    with pytest.raises(RuntimeError, match="tokens are its only actions"):
        validate_hidden_carry_rollout(gated)
    thinking = copy.deepcopy(batch)
    thinking.actions[0, batch.prompt_length - 1] = 0
    with pytest.raises(RuntimeError, match="tokens are its only actions"):
        validate_hidden_carry_rollout(thinking)


def test_continuous_refill_refuses_the_hidden_carry():
    with pytest.raises(ValueError, match="does not implement the hidden carry"):
        rollout_continuous_refill_groups(
            _carry_wrapper(),
            [_prompts("nano", 1, 4)],
            [torch.tensor([4])],
            prompt_repeats=2,
            capacity_rows=2,
            max_new_tokens=2,
            max_stream_steps=2,
            temperature=1.0,
            top_p=1.0,
            seed=1,
            cache_dtype=torch.float32,
            pin_emit=True,
        )


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--thought-sigma", "1.0"], "--thought-sigma configures"),
        (
            ["--init-stop-thinking-probability", "0.9"],
            "--init-stop-thinking-probability configures",
        ),
        (
            ["--rollout-scheduler", "continuous_refill"],
            "requires --rollout-scheduler lockstep",
        ),
    ],
)
def test_carry_cli_refuses_stochastic_and_refill_flags(extra, message, capsys):
    parser = build_arg_parser()
    args = parser.parse_args(
        ["--checkpoint", "c", "--output", "o", "--reasoning-mode", "carry", *extra]
    )
    with pytest.raises(SystemExit):
        validate_args(parser, args)
    assert message in capsys.readouterr().err


def test_cli_fills_legacy_stochastic_defaults_outside_carry():
    """cot/none/latent keep the 1.0/0.9 resume fields; carry records None.

    Outside carry an explicit value is still accepted, exactly as before, so
    a run that recorded one stays resumable.
    """
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--checkpoint", "c", "--output", "o", "--reasoning-mode", "cot",
            "--thought-sigma", "0.5", "--init-stop-thinking-probability", "0.8",
        ]
    )
    validate_args(parser, args)
    assert (args.thought_sigma, args.init_stop_thinking_probability) == (0.5, 0.8)
    for mode, expected, extra in (
        ("latent", (1.0, 0.9), ()),
        ("cot", (1.0, 0.9), ()),
        ("none", (1.0, 0.9), ("--no-think-tokens", "--no-answer-fence")),
        ("carry", (None, None), ()),
    ):
        args = parser.parse_args(
            ["--checkpoint", "c", "--output", "o", "--reasoning-mode", mode, *extra]
        )
        validate_args(parser, args)
        assert (
            args.thought_sigma,
            args.init_stop_thinking_probability,
        ) == expected, mode


# --- checkpoints ----------------------------------------------------------------


def test_carry_checkpoint_round_trips_and_resumes_under_its_own_contract(tmp_path):
    wrapper = _carry_wrapper()
    critic = _critic(live=True)
    optimizers = build_optimizers(wrapper, critic, learning_rate=1e-3, fused=False)
    sampler = MathPromptSampler([{"id": index} for index in range(4)], seed=3)
    path = tmp_path / "carry.pt"
    save_checkpoint(
        path,
        wrapper,
        critic,
        optimizers,
        step=0,
        args=_cot_args(
            reasoning_mode="carry",
            thought_sigma=None,
            init_stop_thinking_probability=None,
        ),
        sampler=sampler,
        warmup_step=0,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["execution_schema"] == schemas.CARRY_EXECUTION_SCHEMA
    assert payload["replay_numerics_schema"] == (
        schemas.CARRY_REPLAY_NUMERICS_SCHEMA
    )
    assert payload["actor_objective_schema"] == (
        schemas.CARRY_ACTOR_OBJECTIVE_SCHEMA
    )
    assert payload["rollout_policy_schema"] == HIDDEN_CARRY_ROLLOUT_POLICY_SCHEMA
    assert payload["thought_input_schema"] == HIDDEN_CARRY_INPUT_SCHEMA
    assert payload["thought_distribution_schema"] is None
    assert schemas.resume_execution_schema_compatible(
        payload, expected_execution_schema=schemas.execution_schema_for("carry")
    )
    assert schemas.resume_replay_schema_compatible(payload, "carry")
    assert not schemas.resume_replay_schema_compatible(payload, "cot")

    validate_renderer_checkpoint(
        payload,
        str(path),
        expected_rollout_policy_schema=rollout_policy_schema_for_mode("carry"),
    )
    kwargs = combiner_init_kwargs_from_checkpoint(payload)
    assert kwargs["hidden_carry"] is True
    restored = LatentThoughtModel(_nano_backbone(seed=9), **kwargs).eval()
    restored.load_state_dict(payload["model"], strict=True)
    restored_critic = SeparateCritic(_nano_backbone(seed=10))
    restored_critic.load_state_dict(payload["critic"], strict=True)
    restored_optimizers = build_optimizers(
        restored, restored_critic, learning_rate=1e-3, fused=False
    )
    for name, optimizer in restored_optimizers.items():
        optimizer.load_state_dict(payload["optimizers"][name])

    prompt_ids = _prompts("nano", 4, 5)
    expected = _roll(wrapper, prompt_ids)
    actual = _roll(restored, prompt_ids)
    for field in fields(expected):
        if field.name == "replay_layout_token":
            continue
        left, right = getattr(expected, field.name), getattr(actual, field.name)
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right), field.name
        else:
            assert left == right, field.name

    # A carry wrapper cannot be labelled with another mode's contract.
    with pytest.raises(ValueError, match="does not match a wrapper"):
        save_checkpoint(
            tmp_path / "mislabelled.pt",
            wrapper,
            critic,
            optimizers,
            step=0,
            args=_cot_args(reasoning_mode="cot"),
            sampler=sampler,
            warmup_step=0,
        )


def test_stale_and_foreign_checkpoints_are_refused_as_carry():
    base = {
        "renderer_features_schema": "combined_input+belief/v1",
        "rollout_policy_schema": HIDDEN_CARRY_ROLLOUT_POLICY_SCHEMA,
        "thought_input_schema": HIDDEN_CARRY_INPUT_SCHEMA,
        "thought_distribution_schema": None,
        "args": {},
    }
    expected = rollout_policy_schema_for_mode("carry")
    validate_renderer_checkpoint(base, "ok", expected_rollout_policy_schema=expected)
    for override, message in (
        # The v28 carry: same idea, older stream and objective.
        ({"rollout_policy_schema": "deterministic_hidden_carry/v1"}, "rollout"),
        # A v29 latent-thought checkpoint.
        ({"rollout_policy_schema": ROLLOUT_POLICY_SCHEMA}, "rollout"),
        ({"thought_input_schema": THOUGHT_INPUT_SCHEMA}, "hidden-carry input"),
        (
            {"thought_distribution_schema": THOUGHT_DISTRIBUTION_SCHEMA},
            "samples no thought distribution",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            validate_renderer_checkpoint(
                {**base, **override},
                "stale",
                expected_rollout_policy_schema=expected,
            )
    # Geometry recovery keys off the schema, never key presence: a v1 carry
    # payload is read as a stochastic policy and refused for lacking one.
    with pytest.raises(ValueError, match="stochastic policy arguments"):
        combiner_init_kwargs_from_checkpoint(
            {**base, "rollout_policy_schema": "deterministic_hidden_carry/v1"}
        )
