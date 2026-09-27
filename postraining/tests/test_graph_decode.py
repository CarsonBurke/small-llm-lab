"""``PinnedDecodeArena`` against the eager pinned-EMIT rollout contract.

The arena runs the same decode tick as a static-shape function over bucketed
row views so that CUDA can replay it as one graph. On CPU the tick runs
eagerly, which pins everything except the graph replay and the CUDA
attention kernel (held to its reference in the GPU suites): bucket padding,
survivor compaction, the per-row key range, prompt fan-out, arena reuse and
the counter-keyed sampler must reproduce ``rollout_continuations`` with the
same ``token_seeds`` stream for stream.
"""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from postraining.graph_decode import (
    PinnedDecodeArena,
    autocast_linear_parameter_names,
)
from postraining.latent_rollout import (
    PAD_SLOT,
    _race_winner,
    counter_gumbel_scores,
    counter_gumbel_tokens,
    nucleus_mask,
    replay_beliefs,
    rollout_continuations,
    trajectory_token_seeds,
    trim_stream,
)
from postraining.latent_thought import LatentThoughtModel
from postraining.tests.test_hidden_carry import BACKBONES, _live_combiner

STREAM_FIELDS = ("kind", "token_ids", "actions", "action_mask", "emit_mask")


def _wrapper(kind: str, carry: bool) -> LatentThoughtModel:
    factory, _ = BACKBONES[kind]
    wrapper = LatentThoughtModel(factory(3), hidden_carry=carry).eval()
    if carry:
        _live_combiner(wrapper.combiner, 103)
    return wrapper


def _left_padded(kind: str, lengths: list[int], seed: int = 5):
    _, vocab = BACKBONES[kind]
    generator = torch.Generator().manual_seed(seed)
    width = max(lengths)
    prompt_ids = torch.zeros((len(lengths), width), dtype=torch.long)
    for row, length in enumerate(lengths):
        prompt_ids[row, width - length :] = torch.randint(
            1, vocab, (length,), generator=generator
        )
    return prompt_ids, torch.tensor(lengths)


def _arena(wrapper, carry, rows, kv_width, stop_ids, **kwargs):
    return PinnedDecodeArena(
        wrapper,
        rows=rows,
        kv_width=kv_width,
        temperature=1.0,
        stop_ids=stop_ids,
        device=torch.device("cpu"),
        hidden_carry=carry,
        cache_dtype=torch.float32,
        autocast_dtype=None,
        compile=False,
        cuda_graphs=False,
        **kwargs,
    )


def _eager(
    wrapper, carry, prompt_ids, lengths, repeats, new_tokens, stop_ids, seeds,
    top_p=1.0,
):
    with torch.no_grad():
        return rollout_continuations(
            wrapper,
            prompt_ids,
            new_tokens,
            new_tokens,
            1.0,
            top_p,
            stop_ids=stop_ids,
            prompt_lengths=lengths,
            pin_emit=True,
            hidden_carry=carry,
            record_likelihoods=False,
            prompt_repeats=repeats,
            token_seeds=seeds,
        )


def _assert_same_rollout(expected, actual, carry):
    assert actual.prompt_length == expected.prompt_length
    assert actual.carry_injected == expected.carry_injected == carry
    for name in STREAM_FIELDS:
        assert torch.equal(getattr(expected, name), getattr(actual, name)), name
    if carry:
        torch.testing.assert_close(actual.hiddens, expected.hiddens, rtol=0, atol=1e-5)
    else:
        assert actual.hiddens.size(-1) == expected.hiddens.size(-1) == 0
    for field in fields(expected):
        value = getattr(expected, field.name)
        if isinstance(value, torch.Tensor):
            assert getattr(actual, field.name).shape == value.shape, field.name
            assert getattr(actual, field.name).dtype == value.dtype, field.name


@pytest.mark.parametrize("carry", [False, True])
@pytest.mark.parametrize("kind", sorted(BACKBONES))
def test_arena_rollout_is_the_eager_seeded_rollout(kind, carry, monkeypatch):
    shapes = []
    run = PinnedDecodeArena._run
    monkeypatch.setattr(
        PinnedDecodeArena,
        "_run",
        lambda self, rows: (shapes.append(rows), run(self, rows)),
    )
    wrapper = _wrapper(kind, carry)
    prompt_ids, lengths = _left_padded(kind, [3, 6, 4])
    repeats, new_tokens = 4, 11
    stop_ids = tuple(range(1, 5))
    seeds = torch.randint(0, 2**31, (12,), generator=torch.Generator().manual_seed(9))
    expected = _eager(
        wrapper, carry, prompt_ids, lengths, repeats, new_tokens, stop_ids, seeds
    )
    arena = _arena(
        wrapper,
        carry,
        rows=16,
        kv_width=20,
        stop_ids=stop_ids,
        row_buckets=(16, 12, 8, 4, 2),
        sync_every=2,
    )
    actual = arena.rollout(
        prompt_ids,
        lengths,
        prompt_repeats=repeats,
        max_new_tokens=new_tokens, max_stream_steps=new_tokens,
        token_seeds=seeds,
    )
    _assert_same_rollout(expected, actual, carry)
    # The contract only bites if rows stopped at different times.
    actions = actual.action_mask.sum(1)
    assert len(set(actions.tolist())) > 2
    assert int(actions.min()) < new_tokens
    # Survivors were compacted into a smaller row bucket mid-rollout.
    assert min(shapes) < 12


@pytest.mark.parametrize("kind", sorted(BACKBONES))
def test_key_starts_step_is_the_row_masked_step(kind):
    """A per-row key start is the 2-D key mask it replaces, over the whole
    cache: same logits, same belief, same cache writes."""
    wrapper = _wrapper(kind, carry=False)
    prompt_ids, lengths = _left_padded(kind, [2, 5, 3])
    width, kv_width = prompt_ids.size(1), 12
    pad = width - lengths
    valid = torch.arange(width)[None, :] >= pad[:, None]
    token = torch.tensor([3, 1, 4])
    position = torch.tensor(width)
    keys = torch.arange(kv_width)
    key_mask = (keys[None, :] >= pad[:, None]) & (keys[None, :] <= position)
    outputs = []
    for ranged in (False, True):
        cpu = torch.device("cpu")
        caches = wrapper.make_generation_cache(3, kv_width, cpu)
        prefix = wrapper.make_generation_cache(3, width, cpu)
        with torch.no_grad():
            wrapper.prefill(prompt_ids, prefix, valid)
            for layer, prefix_layer in zip(caches, prefix, strict=True):
                for tensor, source in zip(layer, prefix_layer, strict=True):
                    tensor.zero_()
                    tensor[tuple(slice(0, n) for n in source.shape)].copy_(source)
            step = LatentThoughtModel.step_core(
                wrapper,
                wrapper.embed_tokens(token[:, None]),
                caches,
                position,
                **({"key_starts": pad} if ranged else {"key_mask": key_mask}),
            )
        outputs.append((step, caches))
    (masked, masked_caches), (ranged_step, ranged_caches) = outputs
    for expected, actual in zip(masked, ranged_step, strict=True):
        if expected is None:
            assert actual is None
        else:
            assert torch.equal(expected, actual)
    for expected_layer, actual_layer in zip(masked_caches, ranged_caches, strict=True):
        for expected, actual in zip(expected_layer, actual_layer, strict=True):
            assert torch.equal(expected, actual)


@pytest.mark.parametrize("kind", sorted(BACKBONES))
def test_one_arena_serves_rollouts_of_different_shapes(kind):
    """Reuse must not leak state: a narrower, shorter second rollout agrees."""
    wrapper = _wrapper(kind, carry=False)
    stop_ids = (1, 2, 3)
    arena = _arena(
        wrapper,
        False,
        rows=16,
        kv_width=24,
        stop_ids=stop_ids,
        row_buckets=(16, 8, 4),
        sync_every=3,
    )
    for lengths, repeats, seed in (([5, 2, 7, 3], 4, 1), ([2, 4], 3, 2)):
        prompt_ids, prompt_lengths = _left_padded(kind, lengths, seed=seed)
        batch = len(lengths) * repeats
        seeds = torch.randint(
            0, 2**31, (batch,), generator=torch.Generator().manual_seed(seed)
        )
        expected = _eager(
            wrapper, False, prompt_ids, prompt_lengths, repeats, 9, stop_ids, seeds
        )
        actual = arena.rollout(
            prompt_ids,
            prompt_lengths,
            prompt_repeats=repeats,
            max_new_tokens=9, max_stream_steps=9,
            token_seeds=seeds,
        )
        _assert_same_rollout(expected, actual, carry=False)


@pytest.mark.parametrize("carry", [False, True])
@pytest.mark.parametrize("kind", sorted(BACKBONES))
def test_every_emitted_token_wins_the_race_under_replay_logits(kind, carry):
    """The sampler's key is each row's unpadded stream slot, as replay scores
    it. Parallel replay rounds differently from the recurrent decode, so a
    near-tied race may flip; a wrong key would lose most races outright."""
    wrapper = _wrapper(kind, carry)
    prompt_ids, lengths = _left_padded(kind, [4, 7, 5])
    seeds = trajectory_token_seeds(99, range(3), 3)
    arena = _arena(
        wrapper, carry, rows=9, kv_width=20,
        stop_ids=(1, 2, 3), row_buckets=(9, 6, 3),
        sync_every=2,
    )
    batch = trim_stream(
        arena.rollout(
            prompt_ids, lengths, prompt_repeats=3, max_new_tokens=12, max_stream_steps=12,
            token_seeds=seeds,
        )
    )
    with torch.no_grad():
        stream_inputs, beliefs = replay_beliefs(wrapper, batch)
        logits = wrapper.backbone.logits_from_features(
            wrapper.renderer_features(stream_inputs, beliefs)
        )
    emits = batch.emit_mask.bool()
    pads = (batch.kind != PAD_SLOT).long().argmax(1)
    assert len(set(pads.tolist())) > 1

    def race(key_pads: torch.Tensor) -> tuple[int, int, float]:
        checked = mismatches = 0
        worst = 0.0
        for slot in range(batch.kind.size(1) - 1):
            rows = emits[:, slot].nonzero().squeeze(-1)
            if rows.numel() == 0:
                continue
            scores = counter_gumbel_scores(
                logits[rows, slot], seeds[rows], slot - key_pads[rows], 1.0
            )
            sampled = batch.token_ids[rows, slot + 1]
            winner = scores.argmax(-1)
            wrong = winner != sampled
            if bool(wrong.any()):
                margin = scores.gather(-1, winner[:, None]) - scores.gather(
                    -1, sampled[:, None]
                )
                worst = max(worst, float(margin[wrong].max()))
            checked += rows.numel()
            mismatches += int(wrong.sum())
        return checked, mismatches, worst

    checked, mismatches, worst = race(pads)
    assert checked == int(emits.sum()) > 0
    assert mismatches <= max(1, checked // 20) and worst < 5e-2, (mismatches, worst)
    _, padded_mismatches, _ = race(torch.zeros_like(pads))
    assert padded_mismatches > checked // 4


def test_trajectory_seeds_depend_on_identity_not_chunking():
    whole = trajectory_token_seeds(7, range(6), 4)
    assert whole.shape == (24,) and whole.dtype == torch.int64
    assert len(set(whole.tolist())) == 24
    split = torch.cat(
        [trajectory_token_seeds(7, range(0, 2), 4), trajectory_token_seeds(7, range(2, 6), 4)]
    )
    assert torch.equal(whole, split)
    assert not torch.equal(trajectory_token_seeds(8, range(6), 4), whole)
    # Seeds span the full signed range, high bits included.
    assert bool((whole < 0).any()) or bool((whole >= 2**32).any())


@pytest.mark.parametrize(
    ("kind", "carry", "compile"),
    [
        *((kind, carry, False) for kind in sorted(BACKBONES) for carry in (False, True)),
        # One compiled case (Inductor on CPU is slow): functional_call must
        # trace fullgraph and substitute the copies inside the artifact.
        ("kda", True, True),
    ],
)
def test_cached_rounded_linear_weights_are_bitwise_the_autocast_cast(
    kind, carry, compile
):
    """Rounding linear weights once per rollout changes no bit of a rollout,
    and the cache follows an in-place parameter update."""
    wrapper = _wrapper(kind, carry)
    names = autocast_linear_parameter_names(wrapper)
    assert any(name.startswith("backbone.") for name in names)
    prompt_ids, lengths = _left_padded(kind, [3, 6, 4])
    seeds = trajectory_token_seeds(5, range(3), 4)

    def roll(cache: bool, arena=None):
        arena = arena or PinnedDecodeArena(
            wrapper,
            rows=16,
            kv_width=20,
            temperature=1.0,
            stop_ids=(1, 2, 3, 4),
            device=torch.device("cpu"),
            hidden_carry=carry,
            autocast_dtype=torch.bfloat16,
            row_buckets=(16, 12, 8, 4, 2),
                sync_every=2,
            compile=compile,
            cuda_graphs=False,
            cache_linear_weights=cache,
        )
        batch = arena.rollout(
            prompt_ids, lengths, prompt_repeats=4, max_new_tokens=11, max_stream_steps=11,
            token_seeds=seeds,
        )
        return batch, arena

    cached, arena = roll(True)
    reference, reference_arena = roll(False)
    fields_to_compare = (*STREAM_FIELDS, "hiddens")
    for name in fields_to_compare:
        assert torch.equal(getattr(cached, name), getattr(reference, name)), name
    with torch.no_grad():
        for parameter in wrapper.parameters():
            parameter.mul_(1.25)
    cached, _ = roll(True, arena)
    reference, _ = roll(False, reference_arena)
    for name in fields_to_compare:
        assert torch.equal(getattr(cached, name), getattr(reference, name)), name


def test_budget_exhaustion_without_stop_tokens_runs_every_slot():
    wrapper = _wrapper("nano", carry=False)
    prompt_ids, lengths = _left_padded("nano", [4, 2])
    seeds = torch.arange(4, dtype=torch.long) * 7919
    arena = _arena(
        wrapper, False, rows=4, kv_width=16, stop_ids=(),
        row_buckets=(4, 2), sync_every=5,
    )
    actual = arena.rollout(
        prompt_ids, lengths, prompt_repeats=2, max_new_tokens=8, max_stream_steps=8, token_seeds=seeds
    )
    expected = _eager(wrapper, False, prompt_ids, lengths, 2, 8, None, seeds)
    _assert_same_rollout(expected, actual, carry=False)
    assert bool((actual.action_mask.sum(1) == 8).all())
    assert bool((actual.kind[:, -1] != PAD_SLOT).all())


def test_workspace_lives_only_for_a_rollout(monkeypatch):
    """Caches and carry record are freed after every rollout, even a failed
    one, and a rollout on a re-opened workspace is still the eager one."""
    wrapper = _wrapper("kda", carry=True)
    prompt_ids, lengths = _left_padded("kda", [4, 2, 3])
    seeds = torch.arange(6, dtype=torch.long) * 7919
    arena = _arena(
        wrapper, True, rows=8, kv_width=14, stop_ids=(1, 2),
        row_buckets=(8, 4, 2), sync_every=2,
    )

    def closed() -> bool:
        return arena.caches is None and arena.belief is None and arena.hiddens is None

    assert closed()

    def fail(rows: int) -> None:
        raise RuntimeError("tick failed")

    with monkeypatch.context() as patch:
        patch.setattr(arena, "_run", fail)
        with pytest.raises(RuntimeError, match="tick failed"):
            arena.rollout(
                prompt_ids, lengths, prompt_repeats=2, max_new_tokens=8, max_stream_steps=8,
                token_seeds=seeds,
            )
    assert closed()
    expected = _eager(wrapper, True, prompt_ids, lengths, 2, 8, (1, 2), seeds)
    for _ in range(2):
        actual = arena.rollout(
            prompt_ids, lengths, prompt_repeats=2, max_new_tokens=8, max_stream_steps=8,
            token_seeds=seeds,
        )
        assert closed()
        _assert_same_rollout(expected, actual, carry=True)
    with pytest.raises(RuntimeError, match="workspace"):
        arena._tick_args(8)
    with arena.workspace(arena.belief_dtype):
        with pytest.raises(RuntimeError, match="already open"):
            with arena.workspace(arena.belief_dtype):
                pass
    assert closed()
    with pytest.raises(RuntimeError, match="belief dtype"):
        with arena.workspace(torch.float64):
            pass


def test_compiled_tick_ignores_the_callers_autocast_and_grad_state():
    """Re-capturing per rollout must never recompile: under a CUDA stream
    capture a recompile is an illegal operation. The tick sets its own
    autocast, so the caller's ambient state must not reach its guards."""
    wrapper = _wrapper("kda", carry=False)
    prompt_ids, lengths = _left_padded("kda", [4, 2])
    seeds = torch.arange(4, dtype=torch.long) * 7919
    torch._dynamo.reset()
    arena = PinnedDecodeArena(
        wrapper,
        rows=4,
        kv_width=12,
        temperature=1.0,
        stop_ids=(1,),
        device=torch.device("cpu"),
        hidden_carry=False,
        cache_dtype=torch.float32,
        autocast_dtype=None,
        row_buckets=(4, 2),
        sync_every=2,
        compile=True,
        cuda_graphs=False,
    )

    def rollout():
        return arena.rollout(
            prompt_ids, lengths, prompt_repeats=2, max_new_tokens=6, max_stream_steps=6,
            token_seeds=seeds,
        )

    expected = rollout()
    with torch._dynamo.config.patch(error_on_recompile=True):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            under_autocast = rollout()
        with torch.enable_grad():
            under_grad = rollout()
    for actual in (under_autocast, under_grad):
        _assert_same_rollout(expected, actual, carry=False)


def test_arena_refuses_rollouts_it_cannot_hold():
    wrapper = _wrapper("nano", carry=False)
    arena = _arena(wrapper, False, rows=4, kv_width=10, stop_ids=())
    prompt_ids, lengths = _left_padded("nano", [3, 3])
    seeds = torch.zeros(6, dtype=torch.long)
    with pytest.raises(ValueError, match="exceed the arena"):
        arena.rollout(
            prompt_ids, lengths, prompt_repeats=3, max_new_tokens=6, max_stream_steps=6, token_seeds=seeds
        )
    with pytest.raises(ValueError, match="arena width"):
        arena.rollout(
            prompt_ids, lengths, prompt_repeats=2, max_new_tokens=8, max_stream_steps=8,
            token_seeds=seeds[:4],
        )
    with pytest.raises(ValueError, match="hidden-carry wrapper"):
        _arena(wrapper, True, rows=4, kv_width=10, stop_ids=())


def test_counter_gumbel_samples_the_softmax_distribution():
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0, -3.0, 0.5, 1.5, -0.5])
    rows = 4096
    seeds = torch.randint(0, 2**31, (rows,), generator=torch.Generator().manual_seed(3))
    counts = torch.zeros(logits.numel())
    for slot in range(64):
        tokens = counter_gumbel_tokens(logits.expand(rows, -1), seeds, slot, 1.0)
        counts += torch.bincount(tokens, minlength=logits.numel())
    total = counts.sum()
    expected = logits.softmax(-1) * total
    chi_square = float(((counts - expected) ** 2 / expected).sum())
    # 7 degrees of freedom: the 99.9th percentile is 24.3.
    assert chi_square < 24.3, (counts / total, logits.softmax(-1))
    # Temperature divides the logits before the race.
    hot = torch.zeros(logits.numel())
    for slot in range(16):
        tokens = counter_gumbel_tokens(logits.expand(rows, -1), seeds, slot, 2.0)
        hot += torch.bincount(tokens, minlength=logits.numel())
    expected_hot = (logits / 2.0).softmax(-1) * hot.sum()
    assert float(((hot - expected_hot) ** 2 / expected_hot).sum()) < 24.3


def test_counter_gumbel_draws_depend_only_on_seed_slot_and_logits():
    generator = torch.Generator().manual_seed(4)
    logits = torch.randn(10, 50, generator=generator)
    seeds = torch.randint(0, 2**31, (10,), generator=generator)
    tokens = counter_gumbel_tokens(logits, seeds, 17, 1.0)
    order = torch.randperm(10, generator=generator)
    assert torch.equal(counter_gumbel_tokens(logits[order], seeds[order], 17, 1.0), tokens[order])
    # A 0-dim tensor slot (the graph tick's form) keys the same lanes.
    assert torch.equal(counter_gumbel_tokens(logits, seeds, torch.tensor(17), 1.0), tokens)
    # Different slots and seeds decorrelate.
    assert not torch.equal(counter_gumbel_tokens(logits, seeds, 18, 1.0), tokens)
    # Seeds that agree in their low 32 bits still draw independently.
    high = seeds + (torch.arange(1, 11) << 32)
    noise = counter_gumbel_scores(logits, seeds, 17, 1.0) - logits
    high_noise = counter_gumbel_scores(logits, high, 17, 1.0) - logits
    assert not torch.isclose(noise, high_noise).any(dim=-1).all()
    assert float((noise - high_noise).abs().mean()) > 0.5


def test_counter_gumbel_winner_is_exactly_the_row_argmax():
    """The sliced two-stage reduction keeps argmax's first-index ties."""
    generator = torch.Generator().manual_seed(6)
    vocab = 48 * 25
    for rows in (1, 13):
        scores = torch.randn(rows, vocab, generator=generator)
        coarse = scores.round()
        masked = scores.clone()
        masked[:, : vocab // 2] = -torch.inf
        tied = torch.zeros(rows, vocab)
        tied[:, [70, 700, 1100]] = 3.0
        poisoned = scores.clone()
        poisoned[:, [300, 900]] = torch.nan
        for case in (scores, coarse, masked, tied, poisoned):
            assert torch.equal(_race_winner(case), case.argmax(dim=-1))
        assert _race_winner(tied).eq(70).all()
        compiled = torch.compile(_race_winner, fullgraph=True, dynamic=False)
        for case in (scores, coarse, tied, poisoned):
            assert torch.equal(compiled(case), case.argmax(dim=-1))
    # A vocabulary the slices do not divide falls back to one stage.
    odd = torch.randn(3, 1201, generator=generator)
    assert torch.equal(_race_winner(odd), odd.argmax(dim=-1))


def test_counter_gumbel_lanes_are_uniform_and_never_veto_a_token():
    """The noise is standard Gumbel, finite, and uncorrelated across lanes."""
    logits = torch.zeros(2048, 4096)
    seeds = trajectory_token_seeds(3, range(2048), 1)
    noise = counter_gumbel_scores(logits, seeds, 5, 1.0)
    assert bool(noise.isfinite().all())
    # Gumbel(0, 1): mean is the Euler-Mascheroni constant, variance pi^2 / 6.
    assert abs(float(noise.mean()) - 0.5772) < 5e-3
    assert abs(float(noise.var()) - 1.6449) < 1e-2
    uniform = torch.exp(-torch.exp(-noise.double()))
    counts = torch.histc(uniform.float(), bins=64, min=0.0, max=1.0)
    expected = uniform.numel() / 64
    assert float(((counts - expected) ** 2 / expected).sum()) < 120.0  # 63 dof
    # Neighbouring vocabulary lanes and neighbouring slots are uncorrelated.
    flat = noise - noise.mean()
    lag = float((flat[:, 1:] * flat[:, :-1]).mean() / flat.var())
    next_slot = counter_gumbel_scores(logits, seeds, 6, 1.0)
    next_flat = next_slot - next_slot.mean()
    cross = float((flat * next_flat).mean() / flat.var())
    assert abs(lag) < 5e-3 and abs(cross) < 5e-3
    # The largest 32-bit lanes round to 1.0; the ceiling keeps them finite.
    tops = torch.tensor([2**32 - 1], dtype=torch.float32)
    assert float((tops + 0.5) * (1.0 / 2**32)) == 1.0


@pytest.mark.parametrize("carry", [False, True])
@pytest.mark.parametrize("kind", sorted(BACKBONES))
def test_nucleus_arena_rollout_is_the_eager_seeded_nucleus_rollout(kind, carry):
    """An evaluation arena races over the top-p nucleus exactly as the eager
    seeded rollout does (``counter_gumbel_tokens`` pins the nucleus race
    itself), and the truncation changes what is sampled."""
    wrapper = _wrapper(kind, carry)
    prompt_ids, lengths = _left_padded(kind, [3, 6, 4])
    stop_ids = tuple(range(1, 5))
    seeds = trajectory_token_seeds(21, range(3), 4)
    expected = _eager(
        wrapper, carry, prompt_ids, lengths, 4, 11, stop_ids, seeds, top_p=0.3
    )
    arena = _arena(
        wrapper, carry, rows=16, kv_width=20, stop_ids=stop_ids, top_p=0.3,
        row_buckets=(16, 12, 8, 4, 2), sync_every=2,
    )
    actual = trim_stream(
        arena.rollout(
            prompt_ids, lengths, prompt_repeats=4, max_new_tokens=11,
            max_stream_steps=11, token_seeds=seeds,
        )
    )
    _assert_same_rollout(trim_stream(expected), actual, carry)
    full = trim_stream(
        _eager(wrapper, carry, prompt_ids, lengths, 4, 11, stop_ids, seeds)
    )
    assert not torch.equal(full.token_ids, actual.token_ids)


def test_one_arena_serves_different_token_budgets():
    wrapper = _wrapper("nano", carry=False)
    prompt_ids, lengths = _left_padded("nano", [4, 2])
    seeds = torch.arange(4, dtype=torch.long) * 7919
    arena = _arena(
        wrapper, False, rows=4, kv_width=16, stop_ids=(), row_buckets=(4, 2),
        sync_every=3,
    )
    for budget in (8, 3, 11):
        actual = arena.rollout(
            prompt_ids, lengths, prompt_repeats=2, max_new_tokens=budget,
            max_stream_steps=11, token_seeds=seeds,
        )
        expected = rollout_continuations(
            wrapper, prompt_ids, budget, 11, 1.0, 1.0, prompt_lengths=lengths,
            pin_emit=True, record_likelihoods=False, prompt_repeats=2,
            token_seeds=seeds,
        )
        _assert_same_rollout(expected, actual, carry=False)
        assert bool((actual.action_mask.sum(1) == budget).all())
    with pytest.raises(ValueError, match="fit the token budget"):
        arena.rollout(
            prompt_ids, lengths, prompt_repeats=2, max_new_tokens=12,
            max_stream_steps=11, token_seeds=seeds,
        )


def test_nucleus_mask_is_the_sorted_top_p_rule_with_boundary_ties_kept():
    generator = torch.Generator().manual_seed(8)
    logits = torch.randn(64, 40, generator=generator) * 2.0
    for temperature, top_p in ((1.0, 0.7), (0.5, 0.3), (2.0, 0.95), (1.0, 1e-6)):
        keep = nucleus_mask(logits, temperature, top_p)
        probabilities = (logits / temperature).softmax(-1)
        for row in range(logits.size(0)):
            above = (
                probabilities[row][None, :] > probabilities[row][:, None]
            ).float() @ probabilities[row]
            assert torch.equal(keep[row], above <= top_p + 1e-6), row
        assert bool(keep.any(-1).all())
    # Tied logits at the boundary are all kept, whatever the sort order.
    tied = torch.tensor([[3.0, 1.0, 1.0, 1.0, -2.0]])
    assert nucleus_mask(tied, 1.0, 0.75).tolist() == [[True, True, True, True, False]]
    assert torch.equal(nucleus_mask(logits, 1.0, 1.0), torch.ones_like(keep))


def test_counter_gumbel_nucleus_samples_the_renormalized_nucleus():
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0, -3.0, 0.5, 1.5, -0.5])
    rows = 4096
    seeds = torch.randint(0, 2**31, (rows,), generator=torch.Generator().manual_seed(6))
    keep = nucleus_mask(logits[None], 1.0, 0.8)[0]
    assert 1 < int(keep.sum()) < logits.numel()
    counts = torch.zeros(logits.numel())
    for slot in range(64):
        tokens = counter_gumbel_tokens(logits.expand(rows, -1), seeds, slot, 1.0, 0.8)
        counts += torch.bincount(tokens, minlength=logits.numel())
    assert float(counts[~keep].sum()) == 0.0
    expected = logits[keep].softmax(-1) * counts.sum()
    chi_square = float(((counts[keep] - expected) ** 2 / expected).sum())
    # keep.sum() - 1 <= 6 degrees of freedom: the 99.9th percentile is 22.5.
    assert chi_square < 22.5, (counts, expected)
    with pytest.raises(ValueError, match="top_p"):
        counter_gumbel_tokens(logits[None], seeds[:1], 0, 1.0, 0.0)


class _DigitTokenizer:
    """Token ids decode to digits, so graded answers depend on the draw."""

    def eos_id(self) -> int:
        return 1

    def bos_id(self) -> int:
        return 2

    def encode(self, text: str) -> list[int]:
        return [3 + (ord(char) % 20) for char in text]

    def decode(self, ids: list[int]) -> str:
        return "Answer: " + "".join(str(token % 10) for token in ids[-1:])

    def id_to_piece(self, token: int) -> str:
        return str(token)


def test_arena_evaluation_is_invariant_to_its_batching():
    """Keyed per (seed, row, sample), an arena evaluation samples the same
    trajectories whatever the arena's row count, and refuses an arena whose
    sampling contract differs from the evaluation's."""
    from postraining.latent_eval import evaluate_latent_math

    wrapper = _wrapper("nano", carry=False)
    rows = [
        {"prompt": [{"content": text}], "reward_model": {"ground_truth": str(n % 10)}}
        for n, text in enumerate(("a", "abcd", "ab", "abcdef", "abc"))
    ]

    def evaluate(arena_rows: int, top_p: float = 0.8):
        arena = _arena(
            wrapper, False, rows=arena_rows, kv_width=24, stop_ids=(1, 2),
            top_p=top_p, row_buckets=(16, 8, 4, 2), sync_every=2,
        )
        attempts: list[dict] = []
        metrics = evaluate_latent_math(
            wrapper, _DigitTokenizer(), rows, samples=4, max_new_tokens=10,
            max_stream_steps=10, chunk=4, seed=11, device=torch.device("cpu"),
            prompt_tokens=12, captured_attempts=attempts, top_p=0.8,
            pin_emit=True, decode_arena=arena,
        )
        return metrics, attempts

    wide, wide_attempts = evaluate(16)
    narrow, narrow_attempts = evaluate(4)
    assert wide["sampling_schema"] == "counter_gumbel_row_sample_keys/v1"
    assert wide["samples"] == narrow["samples"] == 20
    for key in ("prompt_correct_counts", "emitted_tokens_mean", "ended_fraction"):
        assert wide[key] == narrow[key], key
    assert [a["emitted_token_ids"] for a in wide_attempts] == [
        a["emitted_token_ids"] for a in narrow_attempts
    ]
    assert len({tuple(a["emitted_token_ids"]) for a in wide_attempts}) > 4
    with pytest.raises(ValueError, match="top_p differs"):
        evaluate(16, top_p=0.5)
