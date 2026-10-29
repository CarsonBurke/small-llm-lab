"""Gated-delta memory carry (``--reasoning-mode memory``) invariants.

The memory carry is a token-only policy whose carried beliefs -- every prompt
belief and every belief that produced a generated token -- are written into a
per-row fp32 gated-delta memory that each generated token's input reads
through a zero-initialized projection. These tests pin, on the nano (RoPE)
and KDA (recurrent) trunks: the scan's definition and bound, bitwise-cot at
initialization, rollout/replay parity with a live memory (prompt writes
included), causality of the record for actor and critic, row alignment
through left padding and packing, the CUDA-graph arena against the eager
loop, gradient through the memory's own history, the critic reading the
actor's memory without training it, and the refusals that keep a memory
stream from being read as any other policy.
"""

from __future__ import annotations

import copy
import math
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from postraining.graph_decode import PinnedDecodeArena
from postraining.latent_rollout import (
    PAD_SLOT,
    assign_terminal_rewards,
    compact_stream_to_device,
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
    MEMORY_CARRY_INPUT_SCHEMA,
    MEMORY_CARRY_ROLLOUT_POLICY_SCHEMA,
    LatentThoughtModel,
    combiner_init_kwargs_from_checkpoint,
    rollout_policy_schema_for_mode,
    validate_renderer_checkpoint,
)
from postraining.memory_carry import (
    MEMORY_MAX_TIMESCALE,
    MEMORY_MIN_TIMESCALE,
    GatedDeltaMemory,
    delta_read,
    delta_write,
    initial_decay_bias,
    memory_write_mask,
    reference_delta_scan,
)
from postraining.reasoning_modes import (
    mode_carries_hidden,
    mode_carries_memory,
    mode_rollout_budget,
)
from postraining.tests.test_hidden_carry import (
    BACKBONES,
    PARITY_TOLERANCE,
    _carry_wrapper,
    _cot_args,
    _nano_backbone,
    _prompts,
    _replayed_token_logprobs,
    _roll,
    _zeroed_hiddens,
)
from postraining.train_latent_vapo import (
    MathPromptSampler,
    aggregate_actor_tensorboard_metrics,
    build_optimizers,
    combiner_weight_telemetry,
    save_checkpoint,
    update_minibatch,
    validate_combiner_geometry,
    validate_critic_only_init,
)
from postraining.value_model import SeparateCritic
from postraining.vapo import schemas
from postraining.vapo.config import build_arg_parser, validate_args

HEADS, HEAD_DIM = 2, 8


def _live_memory(memory: GatedDeltaMemory, seed: int) -> None:
    """Give every memory path real weights, so the record is load-bearing."""
    generator = torch.Generator().manual_seed(seed)

    def draw(parameter, std):
        parameter.copy_(torch.randn(parameter.shape, generator=generator) * std)

    with torch.no_grad():
        draw(memory.reader.out.weight, 0.5 / memory.reader.out.weight.size(1) ** 0.5)
        draw(memory.writer.write_gate.weight, 0.05)
        draw(memory.writer.decay_gate.weight, 0.05)


def _memory_wrapper(kind: str = "nano", *, live: bool = True, seed: int = 3):
    factory, _ = BACKBONES[kind]
    wrapper = LatentThoughtModel(
        factory(seed),
        hidden_carry=True,
        memory_carry=True,
        memory_heads=HEADS,
        memory_head_dim=HEAD_DIM,
    ).eval()
    if live:
        _live_memory(wrapper.combiner, seed + 100)
    return wrapper


def _memory_critic(wrapper, kind: str = "nano", *, live: bool = False, seed: int = 11):
    factory, _ = BACKBONES[kind]
    critic = SeparateCritic(factory(seed), actor_memory=wrapper.combiner).eval()
    with torch.no_grad():
        critic.head.weight.normal_(std=0.05)
        if live:
            critic.combiner.out.weight.normal_(std=0.1)
    return critic


def _random_terms(batch, length, heads, dim, seed):
    generator = torch.Generator().manual_seed(seed)
    unit = lambda x: x / x.norm(dim=-1, keepdim=True)
    q = unit(torch.randn(batch, length, heads, dim, generator=generator))
    k = unit(torch.randn(batch, length, heads, dim, generator=generator))
    v = torch.randn(batch, length, heads, dim, generator=generator)
    log_decay = torch.nn.functional.logsigmoid(
        2.0 + torch.randn(batch, length, heads, generator=generator)
    )
    beta = torch.rand(batch, length, heads, generator=generator)
    return q, k, v, log_decay, beta


# --- the memory itself ---------------------------------------------------------


def test_scan_writes_then_reads_and_masked_slots_are_exact_noops():
    q, k, v, log_decay, beta = _random_terms(3, 9, 2, 4, seed=0)
    reads, final = reference_delta_scan(q, k, v, log_decay, beta)
    state = torch.zeros(3, 2, 4, 4)
    for t in range(9):
        state = delta_write(state, k[:, t], v[:, t], log_decay[:, t], beta[:, t])
        # A slot reads the state that already holds its own write.
        torch.testing.assert_close(reads[:, t], delta_read(state, q[:, t]))
    torch.testing.assert_close(final, state)

    # A zero decay log and zero strength leave the state bitwise untouched.
    masked_decay, masked_beta = log_decay.clone(), beta.clone()
    masked_decay[:, 4] = 0.0
    masked_beta[:, 4] = 0.0
    masked_reads, _ = reference_delta_scan(q, k, v, masked_decay, masked_beta)
    torch.testing.assert_close(
        masked_reads[:, 4], delta_read(_state_after(k, v, log_decay, beta, 4), q[:, 4])
    )

    # The write is the delta rule: full strength, no decay stores v at k.
    stored = delta_write(
        torch.zeros(1, 1, 4, 4), k[:1, 0, :1], v[:1, 0, :1],
        torch.zeros(1, 1), torch.ones(1, 1),
    )
    torch.testing.assert_close(delta_read(stored, k[:1, 0, :1]), v[:1, 0, :1])


def _state_after(k, v, log_decay, beta, length):
    state = torch.zeros(k.size(0), k.size(2), k.size(3), v.size(3))
    for t in range(length):
        state = delta_write(state, k[:, t], v[:, t], log_decay[:, t], beta[:, t])
    return state


def test_state_is_bounded_by_decay_whatever_the_trajectory_length():
    """||S_t|| <= alpha ||S_{t-1}|| + beta |v|: the read cannot run away."""
    q, k, v, log_decay, beta = _random_terms(2, 400, 2, 4, seed=1)
    state = torch.zeros(2, 2, 4, 4)
    alpha_max = float(log_decay.exp().max())
    bound = float((beta[..., None] * v).norm(dim=-1).max()) / (1.0 - alpha_max)
    for t in range(400):
        previous = torch.linalg.matrix_norm(state, ord=2)
        state = delta_write(state, k[:, t], v[:, t], log_decay[:, t], beta[:, t])
        step_bound = (
            log_decay[:, t].exp() * previous
            + beta[:, t] * v[:, t].norm(dim=-1)
        )
        assert bool((torch.linalg.matrix_norm(state, ord=2) <= step_bound + 1e-5).all())
    assert float(torch.linalg.matrix_norm(state, ord=2).max()) <= bound


def test_initial_timescales_are_geometric_across_heads():
    alpha = torch.sigmoid(initial_decay_bias(4))
    timescales = 1.0 / (1.0 - alpha)
    torch.testing.assert_close(timescales[0], torch.tensor(MEMORY_MIN_TIMESCALE))
    torch.testing.assert_close(timescales[-1], torch.tensor(MEMORY_MAX_TIMESCALE))
    ratios = timescales[1:] / timescales[:-1]
    # fp32 sigmoid near 1 costs ~1e-5 relative on 1 / (1 - alpha).
    torch.testing.assert_close(
        ratios, ratios[:1].expand_as(ratios), rtol=1e-4, atol=0
    )


def test_prompt_state_ignores_whatever_pads_hold():
    memory = GatedDeltaMemory(16, heads=HEADS, head_dim=HEAD_DIM)
    _live_memory(memory, seed=4)
    beliefs = torch.randn(2, 6, 16)
    valid = torch.tensor([[False, False, True, True, True, True]] * 2)
    clean = memory.prompt_state(beliefs, valid)
    beliefs[:, :2] = float("nan")
    assert torch.equal(memory.prompt_state(beliefs, valid), clean)


def test_write_mask_needs_a_real_belief_behind_the_slot():
    kind = torch.tensor([[PAD_SLOT, 0, 0, 0, PAD_SLOT], [0, 0, 0, 0, 0]])
    assert memory_write_mask(kind, PAD_SLOT).tolist() == [
        [False, False, True, True, False],
        [False, True, True, True, True],
    ]


# --- modes, schemas, construction ------------------------------------------------


def test_memory_mode_flags_and_schemas_are_its_own():
    assert mode_carries_hidden("memory") and mode_carries_memory("memory")
    assert not mode_carries_memory("carry")
    assert mode_rollout_budget("memory", 700, answer_tokens=8) == (700, 700)
    assert rollout_policy_schema_for_mode("memory") == (
        MEMORY_CARRY_ROLLOUT_POLICY_SCHEMA
    )
    for getter in (
        lambda mode: schemas.execution_schema_for(mode),
        schemas.replay_numerics_schema_for,
        schemas.actor_objective_schema,
        lambda mode: schemas.actor_objective_schema(mode, True),
        lambda mode: schemas.actor_objective_schema(mode, False, True),
    ):
        values = {mode: getter(mode) for mode in ("latent", "carry", "memory", "cot")}
        assert values["memory"] not in {
            value for mode, value in values.items() if mode != "memory"
        }
    with pytest.raises(ValueError, match="lockstep"):
        schemas.execution_schema_for("memory", "continuous_refill")


def test_memory_wrapper_builds_only_the_memory_as_new_parameters():
    wrapper = _memory_wrapper(live=False)
    assert wrapper.gate is None and wrapper.transition is None
    assert isinstance(wrapper.combiner, GatedDeltaMemory)
    assert wrapper.thought_input_schema == MEMORY_CARRY_INPUT_SCHEMA
    assert wrapper.thought_distribution_schema is None
    new = {id(p) for p in wrapper.new_parameters()}
    assert new == {id(p) for p in wrapper.combiner.parameters()}
    # The read projection is the only zero-initialized weight.
    assert float(wrapper.combiner.reader.out.weight.abs().sum()) == 0.0
    assert float(wrapper.combiner.writer.key.weight.abs().sum()) > 0.0
    with pytest.raises(ValueError, match="hidden-carry policy"):
        LatentThoughtModel(
            _nano_backbone(), memory_carry=True, memory_heads=1, memory_head_dim=1
        )
    with pytest.raises(ValueError, match="memory carry only"):
        LatentThoughtModel(_nano_backbone(), hidden_carry=True, memory_heads=2)
    with pytest.raises(ValueError, match="no combined MLP"):
        LatentThoughtModel(
            _nano_backbone(), hidden_carry=True, memory_carry=True,
            memory_heads=1, memory_head_dim=1, mlp_hidden=8,
        )
    with pytest.raises(RuntimeError, match="memory_input"):
        wrapper.combined_input(torch.zeros(1, dtype=torch.long), torch.zeros(1, 256))


# --- rollout and replay ----------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(BACKBONES))
def test_zero_init_memory_rollout_is_bitwise_the_cot_rollout(kind):
    factory, _ = BACKBONES[kind]
    backbone = factory(3)
    cot = LatentThoughtModel(backbone).eval()
    memory = LatentThoughtModel(
        backbone, hidden_carry=True, memory_carry=True,
        memory_heads=HEADS, memory_head_dim=HEAD_DIM,
    ).eval()
    prompt_ids = _prompts(kind, 6, 5)
    expected = _roll(cot, prompt_ids, hidden_carry=False)
    actual = _roll(memory, prompt_ids)
    assert actual.carry_injected and actual.memory_carry
    for field in fields(expected):
        if field.name in (
            "hiddens", "carry_injected", "memory_carry", "replay_layout_token"
        ):
            continue
        left, right = getattr(expected, field.name), getattr(actual, field.name)
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right), field.name
        else:
            assert left == right, field.name
    assert len(set(actual.action_mask.sum(1).tolist())) > 1
    validate_hidden_carry_rollout(actual)
    # The record holds a belief at every slot with a real slot before it --
    # prompt writes included -- and nothing anywhere else.
    written = memory_write_mask(actual.kind, PAD_SLOT)
    assert bool((actual.hiddens[written].abs().sum(-1) > 0).all())
    assert float(actual.hiddens[~written].abs().sum()) == 0.0


@pytest.mark.parametrize("kind", sorted(BACKBONES))
def test_live_memory_rollout_replays_exactly(kind):
    wrapper = _memory_wrapper(kind)
    batch = _roll(wrapper, _prompts(kind, 6, 5))
    tolerance = PARITY_TOLERANCE[kind]
    with torch.no_grad():
        replayed, beliefs = _replayed_token_logprobs(wrapper, batch)
        zeroed, _ = _replayed_token_logprobs(wrapper, _zeroed_hiddens(batch))
    emits = batch.emit_mask.bool()
    torch.testing.assert_close(
        replayed[emits], batch.old_token_logprobs[emits],
        rtol=tolerance, atol=tolerance,
    )
    # hiddens[p] is the belief replay recomputes at p - 1, for prompt and
    # generated slots alike.
    written = memory_write_mask(batch.kind, PAD_SLOT)[:, 1:]
    torch.testing.assert_close(
        batch.hiddens[:, 1:][written], beliefs[:, :-1][written],
        rtol=tolerance, atol=tolerance,
    )
    assert bool(written[:, : batch.prompt_length - 1].all())
    assert (
        (zeroed[emits] - batch.old_token_logprobs[emits]).abs().max()
        > 100 * tolerance
    )


@pytest.mark.parametrize("kind", sorted(BACKBONES))
def test_a_recorded_belief_only_affects_its_own_row_from_its_slot_on(kind):
    wrapper = _memory_wrapper(kind)
    critic = _memory_critic(wrapper, kind, live=True)
    batch = _roll(wrapper, _prompts(kind, 4, 5), stop_ids=())
    for slot in (2, int(generated_slot_mask(batch)[0].nonzero()[3])):
        perturbed = copy.deepcopy(batch)
        perturbed.hiddens[0, slot] += torch.randn(
            batch.hiddens.size(-1), generator=torch.Generator().manual_seed(1)
        )
        with torch.no_grad():
            _, before = replay_beliefs(wrapper, batch)
            _, after = replay_beliefs(wrapper, perturbed)
            values_before = critic.values(batch)
            values_after = critic.values(perturbed)
        # A prompt write reaches the actor only at its first read, but the
        # critic reads at every recorded slot.
        first_read = max(slot, batch.prompt_length)
        for name, left, right, changed in (
            ("actor", before, after, first_read),
            ("critic", values_before, values_after, slot),
        ):
            torch.testing.assert_close(
                right[0, :slot], left[0, :slot], rtol=0, atol=1e-6, msg=name
            )
            torch.testing.assert_close(right[1:], left[1:], rtol=0, atol=1e-6, msg=name)
            assert (right[0, changed] - left[0, changed]).abs().max() > 1e-4, name


def test_memory_alignment_survives_left_padding_split_and_packing():
    wrapper = _memory_wrapper()
    prompts = [
        _prompts("nano", 1, length, seed)[0]
        for length, seed in ((7, 11), (4, 12), (6, 13))
    ]
    samples = 2
    width = max(prompt.numel() for prompt in prompts)
    prompt_ids = torch.zeros((len(prompts) * samples, width), dtype=torch.long)
    for index, prompt in enumerate(prompts):
        prompt_ids[index * samples : (index + 1) * samples, width - prompt.numel() :] = prompt
    prompt_lengths = torch.tensor([p.numel() for p in prompts]).repeat_interleave(samples)
    batched = _roll(
        wrapper, prompt_ids, top_p=1e-6, prompt_lengths=prompt_lengths, trim=False
    )
    groups = [
        trim_stream(group)
        for group in split_rollout_groups(batched, samples, prompt_lengths)
    ]
    for group, prompt in zip(groups, prompts, strict=True):
        expected = _roll(wrapper, prompt[None].expand(samples, -1), top_p=1e-6)
        assert torch.equal(group.token_ids, expected.token_ids)
        assert torch.equal(group.action_mask, expected.action_mask)
        # Left-pad garbage is never recorded: slot 0 has no belief behind it.
        torch.testing.assert_close(group.hiddens, expected.hiddens, rtol=1e-4, atol=1e-5)
        with torch.no_grad():
            replayed, _ = _replayed_token_logprobs(wrapper, group)
        emits = group.emit_mask.bool()
        torch.testing.assert_close(
            replayed[emits], expected.old_token_logprobs[emits], rtol=1e-4, atol=1e-4
        )

    packed = pack_rollout_groups_for_replay(groups)
    assert packed.carry_injected and packed.memory_carry
    critic = _memory_critic(wrapper, live=True)
    with torch.no_grad():
        _, packed_beliefs = replay_beliefs(wrapper, packed)
        packed_values = critic.values(packed)
    row_start = 0
    for group in groups:
        rows = slice(row_start, row_start + samples)
        with torch.no_grad():
            _, beliefs = replay_beliefs(wrapper, group)
            values = critic.values(group)
        torch.testing.assert_close(
            packed_beliefs[rows, : group.stream_length], beliefs, rtol=2e-4, atol=2e-5
        )
        torch.testing.assert_close(
            packed_values[rows, : group.stream_length], values, rtol=2e-4, atol=2e-5
        )
        row_start += samples


@pytest.mark.parametrize("kind", sorted(BACKBONES))
def test_arena_memory_rollout_is_the_eager_seeded_rollout(kind, monkeypatch):
    shapes = []
    run = PinnedDecodeArena._run
    monkeypatch.setattr(
        PinnedDecodeArena, "_run", lambda self, rows: (shapes.append(rows), run(self, rows))
    )
    wrapper = _memory_wrapper(kind)
    _, vocab = BACKBONES[kind]
    generator = torch.Generator().manual_seed(5)
    lengths = [3, 6, 4]
    width = max(lengths)
    prompt_ids = torch.zeros((3, width), dtype=torch.long)
    for row, length in enumerate(lengths):
        prompt_ids[row, width - length :] = torch.randint(1, vocab, (length,), generator=generator)
    lengths = torch.tensor(lengths)
    repeats, new_tokens, stop_ids = 4, 11, tuple(range(1, 5))
    seeds = torch.randint(0, 2**31, (12,), generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        expected = rollout_continuations(
            wrapper, prompt_ids, new_tokens, new_tokens, 1.0, 1.0,
            stop_ids=stop_ids, prompt_lengths=lengths, pin_emit=True,
            hidden_carry=True, record_likelihoods=False,
            prompt_repeats=repeats, token_seeds=seeds,
        )
    arena = PinnedDecodeArena(
        wrapper, rows=16, kv_width=20, temperature=1.0, stop_ids=stop_ids,
        device=torch.device("cpu"), hidden_carry=True,
        cache_dtype=torch.float32, autocast_dtype=None, compile=False,
        cuda_graphs=False, row_buckets=(16, 12, 8, 4, 2), sync_every=2,
    )
    actual = arena.rollout(
        prompt_ids, lengths, prompt_repeats=repeats,
        max_new_tokens=new_tokens, max_stream_steps=new_tokens, token_seeds=seeds,
    )
    assert actual.memory_carry and expected.memory_carry
    for name in ("kind", "token_ids", "actions", "action_mask", "emit_mask"):
        assert torch.equal(getattr(expected, name), getattr(actual, name)), name
    torch.testing.assert_close(actual.hiddens, expected.hiddens, rtol=0, atol=1e-5)
    actions = actual.action_mask.sum(1)
    assert len(set(actions.tolist())) > 2
    assert min(shapes) < 12
    # The arena leaves no state behind for the next rollout to inherit.
    assert arena.memory_state is None


# --- training ----------------------------------------------------------------------


def test_refresh_then_update_is_age_zero_exact_and_trains_through_the_memory():
    wrapper = _memory_wrapper(live=False)
    critic = _memory_critic(wrapper)
    batch = _roll(wrapper, _prompts("nano", 4, 5))
    assign_terminal_rewards(batch, torch.tensor([1.0, 0.0, 1.0, 0.0]))
    refresh_old_statistics(wrapper, critic, batch)
    optimizers = build_optimizers(wrapper, critic, learning_rate=1e-3, fused=False)
    memory = wrapper.combiner
    writer_before = copy.deepcopy(memory.writer.state_dict())
    query_before = memory.reader.query.weight.clone()
    metrics = update_minibatch(wrapper, critic, batch, optimizers)
    assert metrics["token_abs_log_ratio_max"] == 0.0
    assert metrics["carry_input_delta_rms_ratio"] == 0.0
    assert 0.0 < metrics["memory_write_beta_mean"] < 1.0
    # Zero gate weights: every head at its initial timescale, so the
    # log-timescale mean is the log of their geometric mean.
    assert metrics["memory_decay_log_timescale_mean"] == pytest.approx(
        0.5 * math.log(MEMORY_MIN_TIMESCALE * MEMORY_MAX_TIMESCALE), rel=1e-4
    )
    # Zero output: only the read projection has a first-step gradient.
    assert float(memory.reader.out.weight.abs().sum()) > 0.0
    assert torch.equal(memory.reader.query.weight, query_before)
    for name, value in memory.writer.state_dict().items():
        assert torch.equal(value, writer_before[name]), name

    # Second update: now the loss at every read differentiates through every
    # earlier write -- the memory trains its own history.
    batch = _roll(wrapper, _prompts("nano", 4, 5, seed=6))
    assign_terminal_rewards(batch, torch.tensor([1.0, 0.0, 0.0, 1.0]))
    refresh_old_statistics(wrapper, critic, batch)
    metrics = update_minibatch(wrapper, critic, batch, optimizers)
    assert metrics["token_abs_log_ratio_max"] == 0.0
    assert metrics["carry_input_delta_rms_ratio"] > 0.0
    assert not torch.equal(memory.reader.query.weight, query_before)
    for name, value in memory.writer.state_dict().items():
        assert not torch.equal(value, writer_before[name]), name
    dashboard = aggregate_actor_tensorboard_metrics([metrics])
    assert dashboard["memory/write_beta_mean"] == pytest.approx(
        metrics["memory_write_beta_mean"]
    )
    assert "memory/decay_log_timescale_mean" in dashboard
    telemetry = combiner_weight_telemetry(wrapper)
    assert float(telemetry["memory/read_out_weight_rms"]) > 0.0


def test_weight_telemetry_covers_the_critics_memory_inputs():
    """The drill's critic: a memory reader plus twin belief-reading Q heads."""
    wrapper = _memory_wrapper()
    factory, _ = BACKBONES["nano"]
    critic = SeparateCritic(
        factory(11), actor_memory=wrapper.combiner, q_head_hidden=16,
        q_head_reads_actor_belief=True, q_twin=True,
    )
    with torch.no_grad():
        critic.combiner.out.weight.fill_(0.5)
    telemetry = combiner_weight_telemetry(wrapper, critic)
    assert float(telemetry["memory/critic_read_out_weight_rms"]) == pytest.approx(0.5)
    assert float(telemetry["memory/critic_query_weight_rms"]) > 0.0
    assert "memory/critic_q_actor_belief_weight_rms" in telemetry


def test_packed_value_update_is_the_per_group_accumulation():
    """The critic warmup fits its pool as one packed minibatch.

    It used to accumulate one value-only update per prompt group against the
    pool's total action count; packing may change only reduction order.
    """
    wrapper = _memory_wrapper()
    critic = _memory_critic(wrapper, live=True)
    groups = [
        _roll(wrapper, _prompts("nano", 3, 5, seed=seed), seed=seed)
        for seed in (4, 9)
    ]
    for group, rewards in zip(groups, ([1.0, 0.0, 1.0], [0.0, 0.0, 1.0])):
        assign_terminal_rewards(group, torch.tensor(rewards))
    optimizers = build_optimizers(wrapper, critic, learning_rate=1e-3, fused=False)

    def critic_gradients(batches, denominator=None):
        critic.zero_grad(set_to_none=True)
        for batch in batches:
            update_minibatch(
                wrapper, critic, batch, optimizers, value_only=True,
                critic_step=False, value_action_denominator=denominator,
            )
        return [p.grad.clone() for p in critic.parameters() if p.grad is not None]

    total = sum(group.action_mask.sum() for group in groups)
    per_group = critic_gradients(groups, total)
    packed = critic_gradients([pack_rollout_groups_for_replay(groups)])
    assert len(per_group) == len(packed) > 0
    for expected, actual in zip(per_group, packed, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


def test_writer_gradient_comes_from_reads_after_the_write():
    """BPTT inside the memory: a read's loss reaches only earlier writes."""
    wrapper = _memory_wrapper()
    batch = _roll(wrapper, _prompts("nano", 2, 5), stop_ids=())
    generated = generated_slot_mask(batch)
    last = int(generated[0].nonzero()[-1])
    hiddens = batch.hiddens.clone().requires_grad_(False)
    memory = wrapper.combiner
    base = wrapper.embed_tokens(batch.token_ids).detach()
    write_mask = memory_write_mask(batch.kind, PAD_SLOT)
    inputs = memory.stream_inputs(base, hiddens, write_mask, generated)
    inputs[0, last].sum().backward()
    assert memory.writer.value.weight.grad.abs().sum() > 0
    assert memory.writer.decay_gate.bias.grad.abs().sum() > 0
    # The record is sampled data: the replay never differentiates into it.
    assert hiddens.grad is None
    memory.zero_grad()
    # A read before any generated slot sees nothing it can train.
    inputs = memory.stream_inputs(base, hiddens, write_mask, generated)
    inputs[0, batch.prompt_length - 1].sum().backward()
    assert all(
        parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0
        for parameter in memory.parameters()
    )


def test_critic_reads_the_actor_memory_without_training_it():
    wrapper = _memory_wrapper()
    critic = _memory_critic(wrapper)
    # Borrowed, not owned.
    assert not any(key.startswith("actor_memory") for key in critic.state_dict())
    assert {id(p) for p in critic.parameters()}.isdisjoint(
        id(p) for p in wrapper.parameters()
    )
    batch = _roll(wrapper, _prompts("nano", 4, 5))
    with torch.no_grad():
        # A fresh critic reader is an identity.
        assert torch.equal(critic.values(batch), critic.values(_zeroed_hiddens(batch)))
    assign_terminal_rewards(batch, torch.tensor([1.0, 0.0, 1.0, 0.0]))
    refresh_old_statistics(wrapper, critic, batch)
    actor_before = copy.deepcopy(wrapper.state_dict())
    reader_before = critic.combiner.out.weight.clone()
    update_minibatch(
        wrapper, critic, batch,
        build_optimizers(wrapper, critic, learning_rate=1e-3, fused=False),
        value_only=True,
    )
    assert not torch.equal(critic.combiner.out.weight, reader_before)
    for name, value in wrapper.state_dict().items():
        assert torch.equal(value, actor_before[name]), name
    with torch.no_grad():
        assert not torch.equal(critic.values(batch), critic.values(_zeroed_hiddens(batch)))
    # Neither critic reads the other kind of stream.
    with pytest.raises(ValueError, match="memory carry"):
        SeparateCritic(_nano_backbone(seed=10)).values(batch)
    carry_batch = _roll(_carry_wrapper(), _prompts("nano", 2, 5))
    with pytest.raises(ValueError, match="memory carry"):
        critic.values(carry_batch)


# --- refusals and checkpoints ---------------------------------------------------------


def test_memory_streams_refuse_every_other_policy():
    memory = _memory_wrapper()
    carry = _carry_wrapper()
    memory_batch = _roll(memory, _prompts("nano", 2, 5))
    carry_batch = _roll(carry, _prompts("nano", 2, 5))
    for wrapper, batch in ((carry, memory_batch), (memory, carry_batch)):
        with pytest.raises(ValueError, match="rollout carry does not match"):
            replay_beliefs(wrapper, batch)
    with pytest.raises(ValueError, match="different carries"):
        pack_rollout_groups_for_replay([memory_batch, carry_batch])
    # The token-only arm of a memory policy rolls out, but never replays as it.
    token_only = _roll(memory, _prompts("nano", 2, 5), hidden_carry=False)
    assert not token_only.carry_injected and not token_only.memory_carry
    with pytest.raises(ValueError, match="rollout carry does not match"):
        replay_beliefs(memory, token_only)


def test_critic_only_init_refuses_a_memory_critic():
    """Its reader is fit to the source run's writer, not this run's."""
    args = SimpleNamespace(value_warmup_steps=0, reasoning_mode="memory")
    with pytest.raises(ValueError, match="bound to the source run's actor writer"):
        validate_critic_only_init({}, args, "data")


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--reasoning-mode", "cot", "--memory-heads", "2"], "memory only"),
        (["--reasoning-mode", "memory", "--thought-sigma", "1.0"], "samples no thought"),
        (["--reasoning-mode", "memory", "--memory-head-dim", "0"], "must be positive"),
        (
            ["--reasoning-mode", "memory", "--rollout-scheduler", "continuous_refill"],
            "requires --rollout-scheduler lockstep",
        ),
        (
            ["--reasoning-mode", "memory", "--combined-mlp-hidden", "512"],
            "the memory carry has none",
        ),
    ],
)
def test_memory_cli_refusals(extra, message, capsys):
    parser = build_arg_parser()
    args = parser.parse_args(["--checkpoint", "c", "--output", "o", *extra])
    with pytest.raises(SystemExit):
        validate_args(parser, args)
    assert message in capsys.readouterr().err


def test_memory_cli_fills_its_geometry():
    parser = build_arg_parser()
    args = parser.parse_args(["--checkpoint", "c", "--output", "o", "--reasoning-mode", "memory"])
    validate_args(parser, args)
    assert (args.memory_heads, args.memory_head_dim) == (4, 64)
    assert args.thought_sigma is None and args.init_stop_thinking_probability is None


def test_memory_checkpoint_round_trips_under_its_own_contract(tmp_path):
    wrapper = _memory_wrapper()
    critic = _memory_critic(wrapper, live=True)
    optimizers = build_optimizers(wrapper, critic, learning_rate=1e-3, fused=False)
    sampler = MathPromptSampler([{"id": index} for index in range(4)], seed=3)
    path = tmp_path / "memory.pt"
    args = _cot_args(
        reasoning_mode="memory",
        thought_sigma=None,
        init_stop_thinking_probability=None,
        memory_heads=HEADS,
        memory_head_dim=HEAD_DIM,
    )
    save_checkpoint(
        path, wrapper, critic, optimizers, step=0, args=args,
        sampler=sampler, warmup_step=0,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["execution_schema"] == schemas.MEMORY_EXECUTION_SCHEMA
    assert payload["replay_numerics_schema"] == schemas.MEMORY_REPLAY_NUMERICS_SCHEMA
    assert payload["actor_objective_schema"] == schemas.MEMORY_ACTOR_OBJECTIVE_SCHEMA
    assert payload["rollout_policy_schema"] == MEMORY_CARRY_ROLLOUT_POLICY_SCHEMA
    assert payload["thought_input_schema"] == MEMORY_CARRY_INPUT_SCHEMA
    validate_renderer_checkpoint(
        payload, str(path),
        expected_rollout_policy_schema=rollout_policy_schema_for_mode("memory"),
    )
    with pytest.raises(ValueError, match="rollout schema"):
        validate_renderer_checkpoint(
            payload, str(path),
            expected_rollout_policy_schema=rollout_policy_schema_for_mode("carry"),
        )
    with pytest.raises(ValueError, match="carry input schema"):
        validate_renderer_checkpoint(
            {**payload, "thought_input_schema": HIDDEN_CARRY_INPUT_SCHEMA},
            str(path),
            expected_rollout_policy_schema=MEMORY_CARRY_ROLLOUT_POLICY_SCHEMA,
        )
    kwargs = combiner_init_kwargs_from_checkpoint(payload)
    assert kwargs["memory_carry"] and kwargs["hidden_carry"]
    assert (kwargs["memory_heads"], kwargs["memory_head_dim"]) == (HEADS, HEAD_DIM)
    restored = LatentThoughtModel(_nano_backbone(seed=9), **kwargs).eval()
    restored.load_state_dict(payload["model"], strict=True)
    restored_critic = SeparateCritic(_nano_backbone(seed=10), actor_memory=restored.combiner)
    restored_critic.load_state_dict(payload["critic"], strict=True)
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
    with torch.no_grad():
        assert torch.equal(critic.values(expected), restored_critic.values(actual))
    # Resume and --actor-init check memory geometry, never the MLP geometry
    # a memory combiner does not have.
    validate_combiner_geometry(payload, args, True, path)
    with pytest.raises(ValueError, match="memory geometry"):
        validate_combiner_geometry(
            payload, _cot_args(reasoning_mode="memory", memory_heads=HEADS + 1,
                               memory_head_dim=HEAD_DIM), True, path,
        )
    with pytest.raises(ValueError, match="does not match"):
        validate_combiner_geometry(payload, _cot_args(reasoning_mode="carry"), False, path)
    # A memory checkpoint without its geometry is refused, not guessed.
    stripped = copy.deepcopy(payload)
    del stripped["args"]["memory_heads"]
    with pytest.raises(ValueError, match="memory geometry"):
        combiner_init_kwargs_from_checkpoint(stripped)
    # A memory wrapper cannot be labelled with the plain carry's contract.
    with pytest.raises(ValueError, match="does not match a wrapper"):
        save_checkpoint(
            tmp_path / "mislabelled.pt", wrapper, critic, optimizers, step=0,
            args=_cot_args(reasoning_mode="carry"), sampler=sampler, warmup_step=0,
        )
    assert HIDDEN_CARRY_ROLLOUT_POLICY_SCHEMA != MEMORY_CARRY_ROLLOUT_POLICY_SCHEMA


def test_arena_offload_is_the_compacted_device_rollout():
    """``offload_device`` hands back exactly what compacting a clone gives."""
    wrapper = _memory_wrapper()
    prompt_ids = _prompts("nano", 3, 5)
    lengths = torch.full((3,), prompt_ids.size(1))
    cpu = torch.device("cpu")

    def roll(**offload):
        arena = PinnedDecodeArena(
            wrapper, rows=6, kv_width=16, temperature=1.0, stop_ids=(1, 2),
            device=cpu, hidden_carry=True, cache_dtype=torch.float32,
            autocast_dtype=None, compile=False, cuda_graphs=False,
        )
        return arena.rollout(
            prompt_ids, lengths, prompt_repeats=2, max_new_tokens=10,
            max_stream_steps=10, token_seeds=torch.arange(6), **offload,
        )

    expected = compact_stream_to_device(roll(), cpu)
    offloaded = roll(offload_device=cpu)
    assert offloaded.hiddens.size(-1) > 0
    for field in fields(expected):
        value = getattr(expected, field.name)
        if isinstance(value, torch.Tensor):
            assert torch.equal(getattr(offloaded, field.name), value), field.name


def test_evaluation_arena_decodes_the_same_without_a_replay_record():
    """``replay_storage=False`` drops only the record evaluation never reads."""
    wrapper = _memory_wrapper()
    prompt_ids = _prompts("nano", 3, 5)
    lengths = torch.full((3,), prompt_ids.size(1))
    seeds = torch.arange(6)

    def roll(replay_storage):
        arena = PinnedDecodeArena(
            wrapper, rows=6, kv_width=16, temperature=1.0, stop_ids=(1, 2),
            device=torch.device("cpu"), hidden_carry=True,
            cache_dtype=torch.float32, autocast_dtype=None, compile=False,
            cuda_graphs=False, replay_storage=replay_storage,
        )
        return arena.rollout(
            prompt_ids, lengths, prompt_repeats=2,
            max_new_tokens=10, max_stream_steps=10, token_seeds=seeds,
        )

    stored, unstored = roll(True), roll(False)
    for name in ("kind", "token_ids", "actions", "action_mask"):
        assert torch.equal(getattr(stored, name), getattr(unstored, name)), name
    assert unstored.hiddens.size(-1) == 0 and stored.hiddens.size(-1) > 0
    with pytest.raises(ValueError, match="cannot be replayed"):
        replay_beliefs(wrapper, unstored)
