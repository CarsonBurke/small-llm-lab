"""Counter-keyed write noise for the gated-delta memory carry.

A write-noise rollout explores around the deterministic writer: each
generated belief's write adds ``sigma * eps`` to the writer's
pre-activations, ``eps`` keyed on (the row's noise seed, the belief's
unpadded slot). These tests pin the noise itself (determinism, lane stability,
moments, domain separation, exact antithetics and zero), that a replay
regenerates exactly what decode drew -- through the eager loop, the arena,
left padding, splitting and packing, and for the critic -- that a zero
scale is the noiseless policy, and the refusals that keep noise from being
recorded or replayed without its keys.
"""

from __future__ import annotations

import copy

import pytest
import torch

from postraining.counter_noise import counter_bits, counter_normals
from postraining.graph_decode import PinnedDecodeArena
from postraining.latent_rollout import (
    PAD_SLOT,
    generated_slot_mask,
    pack_rollout_groups_for_replay,
    recorded_write_noise,
    replay_beliefs,
    rollout_continuations,
    split_rollout_groups,
    trim_stream,
)
from postraining.memory_carry import (
    WRITE_NOISE_DOMAIN,
    MemoryWriter,
    memory_write_mask,
    write_noise,
)
from postraining.tests.test_hidden_carry import (
    BACKBONES,
    PARITY_TOLERANCE,
    _carry_wrapper,
    _prompts,
    _replayed_token_logprobs,
)
from postraining.tests.test_memory_carry import (
    HEADS,
    HEAD_DIM,
    _memory_critic,
    _memory_wrapper,
)

SEEDS = torch.tensor(
    [0, 1, -1, 2**62 + 12345, -(2**40) + 7, 987654321987], dtype=torch.int64
)


def _stripped(batch):
    """The same record, replayed as if no write noise had been drawn."""
    stripped = copy.copy(batch)
    stripped.write_noise_seeds = stripped.write_noise_scale = None
    return stripped


def _noisy_roll(
    wrapper, prompt_ids, scales, *, seeds=None, noise_seeds=None, new_tokens=8, **kwargs
):
    """A token-seeded rollout; noise seeds default to independent per row."""
    rows = prompt_ids.size(0) * kwargs.get("prompt_repeats", 1)
    if seeds is None:
        seeds = torch.randint(
            0, 2**62, (rows,), generator=torch.Generator().manual_seed(9)
        )
    if noise_seeds is None:
        noise_seeds = seeds ^ 0x5DEECE66D
    kwargs.setdefault("stop_ids", tuple(range(1, 17)))
    with torch.no_grad():
        batch = rollout_continuations(
            wrapper, prompt_ids, new_tokens, new_tokens, 1.0, 1.0,
            pin_emit=True, hidden_carry=True, token_seeds=seeds,
            write_noise_seeds=None if scales is None else noise_seeds,
            write_noise_scale=(
                None if scales is None
                else torch.as_tensor(scales, dtype=torch.float32)
            ),
            **kwargs,
        )
    return batch


# --- the noise ------------------------------------------------------------------


def test_counter_normals_are_keyed_lane_stable_and_standard():
    slots = torch.tensor([0, 5, 9, 4000, 2, 1])
    first = counter_normals(SEEDS, slots, 64, WRITE_NOISE_DOMAIN)
    assert torch.equal(first, counter_normals(SEEDS, slots, 64, WRITE_NOISE_DOMAIN))
    # Lane i is the same whatever else is drawn.
    assert torch.equal(first[:, :16], counter_normals(SEEDS, slots, 16, WRITE_NOISE_DOMAIN))
    # Slot, either seed half and the domain all move every lane.
    for other in (
        counter_normals(SEEDS, slots + 1, 64, WRITE_NOISE_DOMAIN),
        counter_normals(SEEDS ^ 1, slots, 64, WRITE_NOISE_DOMAIN),
        counter_normals(SEEDS ^ (1 << 40), slots, 64, WRITE_NOISE_DOMAIN),
        counter_normals(SEEDS, slots, 64, (WRITE_NOISE_DOMAIN[0] ^ 1, WRITE_NOISE_DOMAIN[1])),
    ):
        assert bool((other != first).all())
    # Integer and broadcast slots key the same values.
    assert torch.equal(
        counter_normals(SEEDS, 7, 8, WRITE_NOISE_DOMAIN),
        counter_normals(SEEDS, torch.full((6,), 7), 8, WRITE_NOISE_DOMAIN),
    )
    many = counter_normals(
        torch.arange(4096, dtype=torch.int64)[:, None] * 7919,
        torch.arange(64)[None, :],
        32,
        WRITE_NOISE_DOMAIN,
    ).double()
    assert bool(many.isfinite().all()) and float(many.abs().max()) < 5.5
    assert abs(float(many.mean())) < 5e-3
    assert abs(float(many.var()) - 1.0) < 5e-3
    assert abs(float((many**4).mean()) - 3.0) < 3e-2
    # Adjacent lanes, slots and rows are uncorrelated.
    for left, right in (
        (many[..., :-1], many[..., 1:]),
        (many[:, :-1], many[:, 1:]),
        (many[:-1], many[1:]),
    ):
        assert abs(float((left * right).mean())) < 5e-3


def test_write_noise_has_its_own_domain_apart_from_the_token_race():
    lanes = torch.arange(512, dtype=torch.int64)
    token = counter_bits(SEEDS, 17, lanes)
    noise = counter_bits(SEEDS, 17, lanes, WRITE_NOISE_DOMAIN)
    assert not bool((token == noise).any())
    # Nor does any nearby slot of the race reproduce the write stream.
    for slot in range(64):
        assert not bool((counter_bits(SEEDS, slot, lanes) == noise).all(-1).any())


def test_write_noise_is_exactly_antithetic_and_exactly_zero_at_zero_scale():
    slots = torch.tensor([3, 3, 8, 0, 41, 2])
    scale = torch.tensor([0.5, -0.5, 1.0, 0.0, 2.0, -0.25])
    noise = write_noise(SEEDS, slots, scale, HEADS, HEAD_DIM)
    flipped = write_noise(SEEDS, slots, -scale, HEADS, HEAD_DIM)
    assert noise.key.shape == noise.value.shape == (6, HEADS, HEAD_DIM)
    assert noise.decay.shape == noise.write.shape == (6, HEADS)
    for term, opposite in zip(noise, flipped, strict=True):
        assert torch.equal(term, -opposite)
        assert float(term[3].abs().sum()) == 0.0
        assert bool((term[[0, 2, 4, 5]] != 0).all())
    # Scale is a multiplier of one fixed draw.
    unit = write_noise(SEEDS, slots, torch.ones(6), HEADS, HEAD_DIM)
    for term, base in zip(noise, unit, strict=True):
        assert torch.equal(term, base * scale.view(-1, *([1] * (base.dim() - 1))))


def test_zero_noise_is_bitwise_the_noiseless_writer():
    writer = MemoryWriter(24, HEADS, HEAD_DIM)
    with torch.no_grad():
        writer.write_gate.weight.normal_(std=0.1)
        writer.decay_gate.weight.normal_(std=0.1)
    hidden = torch.randn(6, 24, generator=torch.Generator().manual_seed(2))
    zero = write_noise(SEEDS, torch.arange(6), torch.zeros(6), HEADS, HEAD_DIM)
    for left, right in zip(writer(hidden), writer(hidden, zero), strict=True):
        assert torch.equal(left, right)
    noisy = writer(hidden, write_noise(SEEDS, torch.arange(6), torch.ones(6), HEADS, HEAD_DIM))
    k, _, log_decay, beta = noisy
    torch.testing.assert_close(k.norm(dim=-1), torch.ones(6, HEADS))
    assert bool((log_decay < 0).all()) and bool(((beta > 0) & (beta < 1)).all())
    for left, right in zip(writer(hidden), noisy, strict=True):
        assert bool((left != right).any())


# --- decode against replay ------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(BACKBONES))
def test_replay_regenerates_the_write_noise_decode_drew(kind):
    wrapper = _memory_wrapper(kind)
    prompt_ids = _prompts(kind, 6, 5)
    scales = [0.0, 1.0, -1.0, 0.5, 2.0, -0.25]
    # No stops: the first noised write (the first generated token's belief)
    # is first read for the third emitted token.
    batch = trim_stream(_noisy_roll(
        wrapper, prompt_ids, scales, record_likelihoods=True, stop_ids=()
    ))
    assert torch.equal(batch.write_noise_scale, torch.tensor(scales))
    assert batch.write_noise_seeds is not None
    tolerance = PARITY_TOLERANCE[kind]
    with torch.no_grad():
        replayed, beliefs = _replayed_token_logprobs(wrapper, batch)
        stripped, _ = _replayed_token_logprobs(wrapper, _stripped(batch))
    emits = batch.emit_mask.bool()
    torch.testing.assert_close(
        replayed[emits], batch.old_token_logprobs[emits], rtol=tolerance, atol=tolerance
    )
    written = memory_write_mask(batch.kind, PAD_SLOT)[:, 1:]
    torch.testing.assert_close(
        batch.hiddens[:, 1:][written], beliefs[:, :-1][written],
        rtol=tolerance, atol=tolerance,
    )
    # The noise is load-bearing on every noised row, and only on those.
    gap = (stripped - batch.old_token_logprobs).abs() * emits
    assert float(gap[0].max()) <= tolerance
    for row in range(1, 6):
        assert float(gap[row].max()) > 100 * tolerance, row


@pytest.mark.parametrize("kind", sorted(BACKBONES))
def test_a_zero_scale_row_is_the_noiseless_rollout(kind):
    wrapper = _memory_wrapper(kind)
    prompt_ids = _prompts(kind, 4, 5)
    seeds = torch.randint(0, 2**62, (4,), generator=torch.Generator().manual_seed(4))
    roll = lambda scales: _noisy_roll(
        wrapper, prompt_ids, scales, seeds=seeds, record_likelihoods=True,
        stop_ids=(), new_tokens=12,
    )
    noiseless, zero, large = roll(None), roll([0.0] * 4), roll([4.0] * 4)
    assert noiseless.write_noise_scale is None and noiseless.write_noise_seeds is None
    for name in ("kind", "token_ids", "action_mask", "hiddens", "old_token_logprobs"):
        assert torch.equal(getattr(noiseless, name), getattr(zero, name)), name
    # Large noise moves the sampled responses themselves.
    assert not torch.equal(noiseless.token_ids, large.token_ids)


def test_only_generated_beliefs_are_noised():
    wrapper = _memory_wrapper()
    batch = _noisy_roll(wrapper, _prompts("nano", 4, 5), [1.0] * 4, stop_ids=())
    noise = recorded_write_noise(batch, wrapper.combiner.writer)
    noised = noise.value.abs().sum((-2, -1)) > 0
    # hiddens[:, j] is the belief computed at slot j - 1: noised exactly when
    # that slot holds a generated token. Prompt writes -- including the
    # belief that produced the first generated token -- are noiseless.
    producer_generated = torch.zeros_like(noised)
    producer_generated[:, 1:] = generated_slot_mask(batch)[:, :-1]
    expected = memory_write_mask(batch.kind, PAD_SLOT) & producer_generated
    assert torch.equal(noised, expected)
    assert not bool(noised[:, : batch.prompt_length + 1].any())
    assert bool(noised[:, batch.prompt_length + 1 :].any())


def test_write_noise_alignment_survives_left_padding_split_and_packing():
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
    seeds = torch.randint(0, 2**62, (6,), generator=torch.Generator().manual_seed(3))
    scales = [1.0, -1.0, 0.5, 2.0, -0.5, 1.5]
    batched = _noisy_roll(
        wrapper, prompt_ids, scales, seeds=seeds,
        prompt_lengths=prompt_lengths, record_likelihoods=True,
    )
    groups = [
        trim_stream(group)
        for group in split_rollout_groups(batched, samples, prompt_lengths)
    ]
    for index, (group, prompt) in enumerate(zip(groups, prompts, strict=True)):
        rows = slice(index * samples, (index + 1) * samples)
        assert torch.equal(group.write_noise_seeds, seeds[rows] ^ 0x5DEECE66D)
        # The noise is keyed on unpadded slots: the group's own unpadded
        # rollout draws the same noise and the same tokens.
        expected = trim_stream(_noisy_roll(
            wrapper, prompt[None].expand(samples, -1), scales[rows],
            seeds=seeds[rows], record_likelihoods=True,
        ))
        assert torch.equal(group.token_ids, expected.token_ids)
        with torch.no_grad():
            replayed, _ = _replayed_token_logprobs(wrapper, group)
        emits = group.emit_mask.bool()
        torch.testing.assert_close(
            replayed[emits], group.old_token_logprobs[emits], rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            group.old_token_logprobs[emits], expected.old_token_logprobs[emits],
            rtol=1e-4, atol=1e-4,
        )

    packed = pack_rollout_groups_for_replay(groups)
    assert torch.equal(packed.write_noise_seeds, seeds ^ 0x5DEECE66D)
    assert torch.equal(packed.write_noise_scale, torch.tensor(scales))
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


def test_the_critic_reads_the_noisy_writes():
    wrapper = _memory_wrapper()
    critic = _memory_critic(wrapper, live=True)
    prompt_ids = _prompts("nano", 4, 5)
    seeds = torch.randint(0, 2**62, (4,), generator=torch.Generator().manual_seed(8))
    noisy = _noisy_roll(wrapper, prompt_ids, [1.0, -1.0, 2.0, 0.5], seeds=seeds)
    zero = _noisy_roll(wrapper, prompt_ids, [0.0] * 4, seeds=seeds)
    with torch.no_grad():
        assert torch.equal(critic.values(zero), critic.values(_stripped(zero)))
        values = critic.values(noisy)
        stripped = critic.values(_stripped(noisy))
    response = generated_slot_mask(noisy)
    assert float(((values - stripped).abs() * response).amax()) > 1e-3


@pytest.mark.parametrize("kind", sorted(BACKBONES))
def test_arena_write_noise_is_the_eager_seeded_rollout(kind):
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
    scales = torch.linspace(-2.0, 2.0, 12)
    expected = _noisy_roll(
        wrapper, prompt_ids, scales, seeds=seeds, new_tokens=new_tokens,
        stop_ids=stop_ids, prompt_lengths=lengths, record_likelihoods=False,
        prompt_repeats=repeats,
    )
    arena = PinnedDecodeArena(
        wrapper, rows=16, kv_width=20, temperature=1.0, stop_ids=stop_ids,
        device=torch.device("cpu"), hidden_carry=True,
        cache_dtype=torch.float32, autocast_dtype=None, compile=False,
        cuda_graphs=False, row_buckets=(16, 12, 8, 4, 2), sync_every=2,
        write_noise=True,
    )
    actual = arena.rollout(
        prompt_ids, lengths, prompt_repeats=repeats,
        max_new_tokens=new_tokens, max_stream_steps=new_tokens,
        token_seeds=seeds, write_noise_seeds=seeds ^ 0x5DEECE66D,
        write_noise_scale=scales,
    )
    for name in ("kind", "token_ids", "actions", "action_mask", "emit_mask"):
        assert torch.equal(getattr(expected, name), getattr(actual, name)), name
    torch.testing.assert_close(actual.hiddens, expected.hiddens, rtol=0, atol=1e-5)
    # Recorded in prompt order, although compaction reordered the arena.
    assert len(set(actual.action_mask.sum(1).tolist())) > 2
    assert torch.equal(actual.write_noise_seeds, seeds ^ 0x5DEECE66D)
    assert torch.equal(actual.write_noise_scale, scales)
    # The noise reached the trajectories.
    noiseless = _noisy_roll(
        wrapper, prompt_ids, None, seeds=seeds, new_tokens=new_tokens,
        stop_ids=stop_ids, prompt_lengths=lengths, record_likelihoods=False,
        prompt_repeats=repeats,
    )
    assert not torch.equal(noiseless.token_ids, actual.token_ids)


def test_rows_sharing_a_noise_seed_share_the_noise_but_not_the_tokens():
    """The design the antithetic sweep relies on: one noise pattern per
    prompt across independently sampled responses."""
    wrapper = _memory_wrapper()
    prompt_ids = _prompts("nano", 1, 5).expand(4, -1)
    seeds = torch.randint(0, 2**62, (4,), generator=torch.Generator().manual_seed(6))
    shared = torch.full((4,), 12345, dtype=torch.int64)
    batch = trim_stream(_noisy_roll(
        wrapper, prompt_ids, [1.0] * 4, seeds=seeds, noise_seeds=shared,
        stop_ids=(), record_likelihoods=True,
    ))
    noise = recorded_write_noise(batch, wrapper.combiner.writer)
    for term in noise:
        assert torch.equal(term[1:], term[:1].expand_as(term[1:]))
    assert len({tuple(row) for row in batch.token_ids.tolist()}) > 1
    with torch.no_grad():
        replayed, _ = _replayed_token_logprobs(wrapper, batch)
    emits = batch.emit_mask.bool()
    torch.testing.assert_close(
        replayed[emits], batch.old_token_logprobs[emits], rtol=1e-4, atol=1e-4
    )


# --- refusals -------------------------------------------------------------------


def test_write_noise_refusals():
    memory = _memory_wrapper()
    prompt_ids = _prompts("nano", 2, 5)
    with pytest.raises(ValueError, match="seeds and scales together"):
        with torch.no_grad():
            rollout_continuations(
                memory, prompt_ids, 4, 4, 1.0, 1.0, pin_emit=True,
                hidden_carry=True, write_noise_scale=torch.ones(2),
            )
    with pytest.raises(ValueError, match="memory carry"):
        _noisy_roll(_carry_wrapper(), prompt_ids, [1.0, 1.0])
    with pytest.raises(ValueError, match="one fp32 scale per trajectory"):
        _noisy_roll(memory, prompt_ids, [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="one int64 seed per trajectory"):
        _noisy_roll(memory, prompt_ids, [1.0, 1.0], noise_seeds=torch.zeros(2))
    arena_kwargs = dict(
        rows=4, kv_width=12, temperature=1.0, stop_ids=(), device=torch.device("cpu"),
        hidden_carry=True, cache_dtype=torch.float32, autocast_dtype=None,
        compile=False, cuda_graphs=False,
    )
    with pytest.raises(ValueError, match="memory carry"):
        PinnedDecodeArena(_carry_wrapper(), write_noise=True, **arena_kwargs)
    seeds = torch.arange(2, dtype=torch.int64)
    rollout = dict(prompt_repeats=1, max_new_tokens=4, max_stream_steps=4, token_seeds=seeds)
    lengths = torch.full((2,), prompt_ids.size(1))
    with pytest.raises(ValueError, match="and only it does"):
        PinnedDecodeArena(memory, **arena_kwargs).rollout(
            prompt_ids, lengths, write_noise_seeds=seeds,
            write_noise_scale=torch.ones(2), **rollout
        )
    with pytest.raises(ValueError, match="and only it does"):
        PinnedDecodeArena(memory, write_noise=True, **arena_kwargs).rollout(
            prompt_ids, lengths, **rollout
        )
    noisy = _noisy_roll(memory, prompt_ids, [1.0, 1.0])
    unkeyed = copy.copy(noisy)
    unkeyed.write_noise_seeds = None
    with pytest.raises(ValueError, match="seeds and scales together"):
        recorded_write_noise(unkeyed, memory.combiner.writer)
    with pytest.raises(ValueError, match="partially labeled write_noise_seeds"):
        pack_rollout_groups_for_replay([noisy, _stripped(noisy)])
