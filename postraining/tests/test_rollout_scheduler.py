from __future__ import annotations

import torch

import postraining.latent_rollout as ordinary_rollout
import postraining.rollout_scheduler as scheduler
from postraining.latent_rollout import PAD_SLOT, trim_stream
from postraining.tests.test_latent_rollout import _deterministic_wrapper


def _chunk(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    long = torch.randint(1, 32, (5,), generator=generator)
    short = torch.randint(1, 32, (3,), generator=generator)
    prompts = torch.zeros((2, 5), dtype=torch.long)
    prompts[0] = long
    prompts[1, -3:] = short
    return prompts, torch.tensor([5, 3])


def _greedy(logits, *_args, **_kwargs):
    return logits.argmax(-1)


def test_pause_merge_seam_matches_two_unscheduled_deterministic_rollouts(
    monkeypatch,
):
    wrapper = _deterministic_wrapper()
    monkeypatch.setattr(ordinary_rollout, "top_p_sample", _greedy)
    monkeypatch.setattr(scheduler, "top_p_sample", _greedy)
    prompts, lengths = _chunk(11)
    states = []
    for origin_id in range(2):
        state = scheduler.start_decode_state(
            wrapper,
            prompts,
            lengths,
            origin_id=origin_id,
            prompt_repeats=2,
            max_new_tokens=4,
            max_stream_steps=8,
            temperature=1.0,
            top_p=1.0,
            pin_emit=False,
            sync_every=1,
        )
        # Pause on the mandatory thought input itself.  This exercises the
        # widest seam (kind + fp32 latent), not merely a token seam.
        assert not scheduler.advance_decode_state(state, stop_position=5)
        assert state.position == state.slab.base_position == 5
        states.append(state)

    merged = scheduler.merge_aligned_decode_states(*states)
    assert merged.rows == 8
    assert scheduler.advance_decode_state(merged)
    results = merged.ledger.finalize(
        merged.slab, merged.position, cpu=torch.device("cpu")
    )

    expected = trim_stream(
        ordinary_rollout.rollout_continuations(
            wrapper,
            prompts,
            max_new_tokens=4,
            max_stream_steps=8,
            temperature=1.0,
            top_p=1.0,
            prompt_lengths=lengths,
            prompt_repeats=2,
        )
    )
    for origin_id in range(2):
        actual = results[origin_id]
        assert actual.prompt_length == prompts.size(1)
        for name in (
            "kind",
            "token_ids",
            "thoughts",
            "actions",
            "action_mask",
            "stop_mask",
            "emit_mask",
        ):
            torch.testing.assert_close(
                getattr(actual, name),
                getattr(expected, name),
                rtol=0,
                atol=0,
            )


def test_tail_pause_keeps_exact_survivors_and_offloads_dense_prefix(
    monkeypatch,
):
    wrapper = _deterministic_wrapper()
    prompts, lengths = _chunk(13)

    def end_three(logits, *_args, **_kwargs):
        assert logits.size(0) == 4
        return torch.tensor([1, 1, 1, 2])

    monkeypatch.setattr(scheduler, "top_p_sample", end_three)
    state = scheduler.start_decode_state(
        wrapper,
        prompts,
        lengths,
        origin_id=0,
        prompt_repeats=2,
        max_new_tokens=4,
        max_stream_steps=8,
        temperature=1.0,
        top_p=1.0,
        stop_ids=1,
        pin_emit=True,
        sync_every=1,
    )
    assert not scheduler.advance_decode_state(state, park_live_rows=1)
    assert state.rows == 1
    assert state.slab.rows == 1
    assert state.slab.base_position == state.position == prompts.size(1)
    assert state.slab.origin_rows.tolist() == [3]
    assert all(cache[0].size(0) == 1 for cache in state.caches)
    assert len(state.ledger.segments) == 1
    prefix = state.ledger.segments[0]
    assert prefix.values["kind"].device.type == "cpu"
    assert prefix.values["kind"].size(0) == 4
    assert prefix.values["kind"].size(1) == prompts.size(1) + 1


def test_scalar_tail_scheduler_merges_at_aligned_position_and_restores_order(
    monkeypatch,
):
    wrapper = _deterministic_wrapper()
    chunks_and_lengths = [_chunk(17), _chunk(19)]
    call_rows: list[int] = []

    def scripted_tokens(logits, *_args, **_kwargs):
        call_rows.append(logits.size(0))
        if logits.size(0) == 4:
            return torch.tensor([1, 1, 1, 2])
        assert logits.size(0) == 2
        return torch.ones(2, dtype=torch.long)

    monkeypatch.setattr(scheduler, "top_p_sample", scripted_tokens)
    stats = scheduler.TailScheduleStats()
    results = scheduler.rollout_scalar_tail_chunks(
        wrapper,
        [item[0] for item in chunks_and_lengths],
        [item[1] for item in chunks_and_lengths],
        prompt_repeats=2,
        max_new_tokens=4,
        max_stream_steps=8,
        temperature=1.0,
        top_p=1.0,
        tail_rows=1,
        stop_ids=1,
        pin_emit=True,
        sync_every=1,
        schedule_stats=stats,
    )

    # One step in each original B4 chunk, followed by one merged B2 tail.
    assert call_rows == [4, 4, 2]
    assert stats == scheduler.TailScheduleStats(
        decode_steps=3,
        row_steps=10,
        useful_actions=10,
        lockstep_decode_steps=4,
        catchup_decode_steps=1,
        parks=1,
        merges=1,
        merge_events=1,
        cohort_admissions=1,
        cohort_rollovers=1,
        cohort_rows_max=2,
        chunks=2,
    )
    assert stats.metrics()["decode_step_utilization"] == 1.0
    assert stats.metrics()["decode_step_savings_fraction"] == 0.25
    assert len(results) == 2
    for result, (prompts, _) in zip(
        results, chunks_and_lengths, strict=True
    ):
        assert result.prompt_length == prompts.size(1)
        assert result.action_mask.sum(1).tolist() == [1, 1, 1, 2]
        assert result.token_ids[:, : prompts.size(1)].equal(
            prompts.repeat_interleave(2, dim=0)
        )
        # The survivor owns exactly the post-seam token; no other row gained
        # a duplicated or missing action while records were stitched.
        assert result.kind[:3, -1].eq(PAD_SLOT).all()
        assert result.token_ids[3, -1].item() == 1


def test_scalar_tail_scheduler_rejects_different_chunk_widths():
    wrapper = _deterministic_wrapper()
    first, first_lengths = _chunk(23)
    second = first[:, :-1]
    second_lengths = torch.tensor([4, 3])
    try:
        scheduler.rollout_scalar_tail_chunks(
            wrapper,
            [first, second],
            [first_lengths, second_lengths],
            prompt_repeats=2,
            max_new_tokens=2,
            max_stream_steps=4,
            temperature=1.0,
            top_p=1.0,
            pin_emit=True,
        )
    except ValueError as error:
        assert "common prompt width" in str(error)
    else:
        raise AssertionError("different physical prompt widths were accepted")


def test_three_chunk_scheduler_preserves_repeated_pause_merge_pause_ledgers(
    monkeypatch,
):
    wrapper = _deterministic_wrapper()
    chunks_and_lengths = [_chunk(seed) for seed in (29, 31, 37)]
    scripted = iter(
        (
            [1, 1, 1, 2],  # origin 0 -> one parked survivor
            [1, 1, 1, 2],  # origin 1 catches the survivor position
            [1, 2],        # merged tail parks one row at the next boundary
            [1, 1, 1, 2],  # origin 2 advances toward that boundary
            [2],           # origin 2 survivor catches up without ending
            [1, 1],        # final merged pair ends together
        )
    )
    call_rows: list[int] = []

    def scripted_tokens(logits, *_args, **_kwargs):
        values = next(scripted)
        call_rows.append(logits.size(0))
        assert len(values) == logits.size(0)
        return torch.tensor(values)

    monkeypatch.setattr(scheduler, "top_p_sample", scripted_tokens)
    results = scheduler.rollout_scalar_tail_chunks(
        wrapper,
        [item[0] for item in chunks_and_lengths],
        [item[1] for item in chunks_and_lengths],
        prompt_repeats=2,
        max_new_tokens=5,
        max_stream_steps=8,
        temperature=1.0,
        top_p=1.0,
        tail_rows=1,
        stop_ids=1,
        pin_emit=True,
        sync_every=1,
    )

    assert call_rows == [4, 4, 2, 4, 1, 2]
    assert [batch.action_mask.sum(1).tolist() for batch in results] == [
        [1, 1, 1, 2],
        [1, 1, 1, 3],
        [1, 1, 1, 3],
    ]
    for result, (prompts, _) in zip(
        results, chunks_and_lengths, strict=True
    ):
        assert result.token_ids[:, : prompts.size(1)].equal(
            prompts.repeat_interleave(2, dim=0)
        )


def test_cohort_defers_zero_step_merges_until_pool_end(monkeypatch):
    wrapper = _deterministic_wrapper()
    chunks_and_lengths = [_chunk(seed) for seed in (38, 39, 40)]
    scripted = iter(
        (
            [1, 1, 1, 2],
            [1, 1, 1, 2],
            [1, 1, 1, 2],
            [1, 1, 1],
        )
    )
    call_rows: list[int] = []

    def scripted_tokens(logits, *_args, **_kwargs):
        values = next(scripted)
        call_rows.append(logits.size(0))
        assert len(values) == logits.size(0)
        return torch.tensor(values)

    monkeypatch.setattr(scheduler, "top_p_sample", scripted_tokens)
    stats = scheduler.TailScheduleStats()
    results = scheduler.rollout_scalar_tail_chunks(
        wrapper,
        [item[0] for item in chunks_and_lengths],
        [item[1] for item in chunks_and_lengths],
        prompt_repeats=2,
        max_new_tokens=4,
        max_stream_steps=8,
        temperature=1.0,
        top_p=1.0,
        tail_rows=3,
        stop_ids=1,
        pin_emit=True,
        sync_every=1,
        schedule_stats=stats,
    )

    # The three aligned singleton tails stay separate while their union fits
    # the parked-row budget, then pay one B3 merge and final decode step.
    assert call_rows == [4, 4, 4, 3]
    assert stats == scheduler.TailScheduleStats(
        decode_steps=4,
        row_steps=15,
        useful_actions=15,
        lockstep_decode_steps=6,
        catchup_decode_steps=2,
        parks=1,
        merges=2,
        merge_events=1,
        cohort_admissions=2,
        cohort_rollovers=0,
        cohort_rows_max=3,
        chunks=3,
    )
    assert [batch.action_mask.sum(1).tolist() for batch in results] == [
        [1, 1, 1, 2],
        [1, 1, 1, 2],
        [1, 1, 1, 2],
    ]


def test_bulk_merge_matches_chronological_pairwise_merges(monkeypatch):
    monkeypatch.setattr(scheduler, "top_p_sample", _greedy)

    def paused_states(wrapper):
        states = []
        for origin_id, seed in enumerate((67, 71, 73)):
            prompts, lengths = _chunk(seed)
            state = scheduler.start_decode_state(
                wrapper,
                prompts,
                lengths,
                origin_id=origin_id,
                prompt_repeats=2,
                max_new_tokens=4,
                max_stream_steps=8,
                temperature=1.0,
                top_p=1.0,
                pin_emit=False,
                sync_every=1,
            )
            assert not scheduler.advance_decode_state(
                state, stop_position=prompts.size(1)
            )
            states.append(state)
        return states

    bulk_states = paused_states(_deterministic_wrapper())
    pairwise_states = paused_states(_deterministic_wrapper())
    bulk = scheduler.merge_aligned_decode_states(*bulk_states)
    pairwise = scheduler.merge_aligned_decode_states(
        pairwise_states[0], pairwise_states[1]
    )
    pairwise = scheduler.merge_aligned_decode_states(
        pairwise, pairwise_states[2]
    )

    for name in ("emitted", "ended", "thinking_active"):
        torch.testing.assert_close(
            getattr(bulk, name), getattr(pairwise, name), rtol=0, atol=0
        )
    for name in (
        "belief",
        "predicted",
        "thought_log_sigma",
        "input_latent",
        "logits",
    ):
        torch.testing.assert_close(
            getattr(bulk.output, name),
            getattr(pairwise.output, name),
            rtol=0,
            atol=0,
        )
    for bulk_layer, pairwise_layer in zip(
        bulk.caches, pairwise.caches, strict=True
    ):
        for bulk_cache, pairwise_cache in zip(
            bulk_layer, pairwise_layer, strict=True
        ):
            live_prefix = bulk.position + 1
            torch.testing.assert_close(
                bulk_cache[:, :, :live_prefix],
                pairwise_cache[:, :, :live_prefix],
                rtol=0,
                atol=0,
            )

    assert scheduler.advance_decode_state(bulk)
    assert scheduler.advance_decode_state(pairwise)
    bulk_results = bulk.ledger.finalize(
        bulk.slab, bulk.position, cpu=torch.device("cpu")
    )
    pairwise_results = pairwise.ledger.finalize(
        pairwise.slab, pairwise.position, cpu=torch.device("cpu")
    )
    for origin_id in range(3):
        for name in (
            "kind",
            "token_ids",
            "thoughts",
            "actions",
            "action_mask",
            "stop_mask",
            "emit_mask",
        ):
            torch.testing.assert_close(
                getattr(bulk_results[origin_id], name),
                getattr(pairwise_results[origin_id], name),
                rtol=0,
                atol=0,
            )


def test_chunk_that_finishes_before_carry_position_does_not_drop_carry(
    monkeypatch,
):
    wrapper = _deterministic_wrapper()
    chunks_and_lengths = [_chunk(41), _chunk(43)]
    scripted = iter(
        (
            [1, 1, 1, 2],  # origin 0 parks one row at position 5
            [1, 1, 1, 1],  # origin 1 ends before it can merge there
            [1],            # parked origin 0 is still resumed and completed
        )
    )
    call_rows: list[int] = []

    def scripted_tokens(logits, *_args, **_kwargs):
        values = next(scripted)
        call_rows.append(logits.size(0))
        assert len(values) == logits.size(0)
        return torch.tensor(values)

    monkeypatch.setattr(scheduler, "top_p_sample", scripted_tokens)
    results = scheduler.rollout_scalar_tail_chunks(
        wrapper,
        [item[0] for item in chunks_and_lengths],
        [item[1] for item in chunks_and_lengths],
        prompt_repeats=2,
        max_new_tokens=4,
        max_stream_steps=8,
        temperature=1.0,
        top_p=1.0,
        tail_rows=1,
        stop_ids=1,
        pin_emit=True,
        sync_every=1,
    )

    assert call_rows == [4, 4, 1]
    assert results[0].action_mask.sum(1).tolist() == [1, 1, 1, 2]
    assert results[1].action_mask.sum(1).tolist() == [1, 1, 1, 1]


def test_schedule_savings_include_lockstep_sync_grid_waste(monkeypatch):
    wrapper = _deterministic_wrapper()
    chunks_and_lengths = [_chunk(47), _chunk(53)]
    scripted = iter(
        (
            [1, 1, 1, 2],
            [1, 1, 1, 2],
            [1, 1, 1, 2],
            [1, 1, 1, 2],
            [1, 1],
            [1, 1],
        )
    )

    def scripted_tokens(logits, *_args, **_kwargs):
        values = next(scripted)
        assert len(values) == logits.size(0)
        return torch.tensor(values)

    monkeypatch.setattr(scheduler, "top_p_sample", scripted_tokens)
    stats = scheduler.TailScheduleStats()
    results = scheduler.rollout_scalar_tail_chunks(
        wrapper,
        [item[0] for item in chunks_and_lengths],
        [item[1] for item in chunks_and_lengths],
        prompt_repeats=2,
        max_new_tokens=5,
        max_stream_steps=8,
        temperature=1.0,
        top_p=1.0,
        tail_rows=1,
        stop_ids=1,
        pin_emit=True,
        sync_every=2,
        schedule_stats=stats,
    )

    assert [batch.action_mask.sum(1).tolist() for batch in results] == [
        [1, 1, 1, 3],
        [1, 1, 1, 3],
    ]
    assert stats.decode_steps == 6
    # Each independent chunk would pay four calls: three useful actions,
    # rounded to the next two-step synchronization boundary.
    assert stats.lockstep_decode_steps == 8
    assert stats.metrics()["decode_step_savings_fraction"] == 0.25


def test_lockstep_counterfactual_never_rounds_past_stream_cap(monkeypatch):
    wrapper = _deterministic_wrapper()
    chunks_and_lengths = [_chunk(59), _chunk(61)]

    def never_stop(logits, *_args, **_kwargs):
        return torch.full(
            (logits.size(0),),
            2,
            dtype=torch.long,
        )

    monkeypatch.setattr(scheduler, "top_p_sample", never_stop)
    stats = scheduler.TailScheduleStats()
    scheduler.rollout_scalar_tail_chunks(
        wrapper,
        [item[0] for item in chunks_and_lengths],
        [item[1] for item in chunks_and_lengths],
        prompt_repeats=2,
        max_new_tokens=5,
        max_stream_steps=5,
        temperature=1.0,
        top_p=1.0,
        tail_rows=1,
        stop_ids=1,
        pin_emit=True,
        sync_every=4,
        schedule_stats=stats,
    )

    assert stats.decode_steps == 10
    assert stats.lockstep_decode_steps == 10
    assert stats.metrics()["decode_step_savings_fraction"] == 0.0
