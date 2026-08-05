from __future__ import annotations

from types import SimpleNamespace

import torch

from postraining.latent_rollout import (
    generated_slot_mask,
    replay_beliefs,
    split_rollout_groups,
)
from postraining.latent_thought import LatentThoughtModel, StepOutput
from postraining.rollout_scheduler import (
    ContinuousScheduleStats,
    _decode_execution_width,
    _top_p_from_uniform,
    rollout_continuous_refill_groups,
    warmup_decode_width_buckets,
)
from postraining.tests.test_kda_backbone import (
    _seeded_backbone as _kda_backbone,
)
from postraining.tests.test_nano_backbone import _backbone


def test_cuda_decode_execution_width_uses_static_main_and_tail_buckets():
    assert _decode_execution_width(497, 512, pending_groups=True) == 512
    assert _decode_execution_width(497, 512, pending_groups=False) == 512
    assert _decode_execution_width(256, 512, pending_groups=False) == 256
    assert _decode_execution_width(255, 512, pending_groups=False) == 256
    assert _decode_execution_width(17, 512, pending_groups=False) == 32
    assert _decode_execution_width(1, 512, pending_groups=False) == 1


class _FakeContinuousModel:
    """CPU contract model whose first prompt token is the target length."""

    def __init__(self, *, stochastic_tokens: bool = False):
        self.backbone = SimpleNamespace(
            tok_emb=SimpleNamespace(embedding_dim=3)
        )
        self.stochastic_tokens = stochastic_tokens
        self.bank_builds = 0
        self.admission_calls: list[list[list[int]]] = []
        self.step_calls: list[tuple[list[int], list[int]]] = []
        self.step_kv_starts: list[list[int]] = []
        self.padded_widths: list[int] = []
        self.cache_allocations = 0

    def make_paged_generation_cache(
        self, capacity, max_length, device, dtype=None
    ):
        del dtype
        self.cache_allocations += 1
        return SimpleNamespace(
            target=torch.zeros(capacity, dtype=torch.long, device=device),
            steps=torch.zeros(capacity, dtype=torch.long, device=device),
            kv_starts=torch.zeros(
                capacity, dtype=torch.long, device=device
            ),
            capacity=capacity,
            max_length=max_length,
        )

    def _output(self, slots, cache):
        rows = slots.numel()
        belief = torch.zeros((rows, 3), dtype=torch.float32)
        logits = torch.zeros((rows, 4), dtype=torch.float32)
        if not self.stochastic_tokens:
            target = cache.target.index_select(0, slots)
            steps = cache.steps.index_select(0, slots)
            stop = steps + 1 >= target
            logits[:, 2] = 40.0
            logits[stop, 1] = 80.0
        return StepOutput(
            belief=belief,
            input_latent=torch.zeros_like(belief),
            logits=logits,
            caches=[],
        )

    def build_prompt_prefix_bank(
        self,
        unique_prompt_ids,
        unique_prompt_lengths,
        *,
        dtype=None,
    ):
        del dtype
        self.bank_builds += 1
        return SimpleNamespace(
            prompt_ids=unique_prompt_ids,
            prompt_lengths=unique_prompt_lengths,
        )

    def admit_prompt_prefixes(
        self,
        bank,
        group_indices,
        group_slot_ids,
        paged_cache,
    ):
        self.admission_calls.append(group_slot_ids.tolist())
        slots = group_slot_ids.flatten()
        selected_prompts = bank.prompt_ids.index_select(0, group_indices)
        selected_lengths = bank.prompt_lengths.index_select(0, group_indices)
        targets = selected_prompts[:, 0].repeat_interleave(
            group_slot_ids.size(1)
        )
        paged_cache.target[slots] = targets
        paged_cache.steps[slots] = 0
        paged_cache.kv_starts[slots] = (
            selected_prompts.size(1) - selected_lengths
        ).repeat_interleave(group_slot_ids.size(1))
        return self._output(slots, paged_cache)

    def paged_step(
        self,
        input_latent,
        paged_cache,
        *,
        slot_ids,
        positions,
        live=None,
    ):
        del input_latent
        if live is not None:
            # Padding rows must be inert. Recording only the live prefix keeps
            # every existing call assertion meaningful while the padded width
            # is still exercised end to end.
            self.padded_widths.append(slot_ids.numel())
            slot_ids = slot_ids[live]
            positions = positions[live]
        self.step_calls.append((slot_ids.tolist(), positions.tolist()))
        self.step_kv_starts.append(
            paged_cache.kv_starts.index_select(0, slot_ids).tolist()
        )
        paged_cache.steps[slot_ids] += 1
        return self._output(slot_ids, paged_cache)

    @staticmethod
    def embed_tokens(token_ids):
        return token_ids.float()[..., None].expand(*token_ids.shape, 3)

    @staticmethod
    def combined_input(token_ids, hidden):
        return token_ids.float()[..., None, None].expand(
            *token_ids.shape, 1, 3
        )


def _chunk(*target_lengths: int):
    prompts = torch.zeros((len(target_lengths), 3), dtype=torch.long)
    prompts[:, 0] = torch.tensor(target_lengths)
    prompts[:, 1:] = 3
    return prompts, torch.full((len(target_lengths),), 3, dtype=torch.long)


def _run(
    model,
    chunks,
    *,
    repeats=2,
    capacity=4,
    max_new_tokens=5,
    seed=17,
    stats=None,
    top_p=1.0,
    pin_emit=True,
    replay_storage=True,
    paged_cache=None,
    pad_decode_width=None,
):
    return rollout_continuous_refill_groups(
        model,
        [chunk[0] for chunk in chunks],
        [chunk[1] for chunk in chunks],
        prompt_repeats=repeats,
        capacity_rows=capacity,
        max_new_tokens=max_new_tokens,
        max_stream_steps=max_new_tokens,
        temperature=1.0,
        top_p=top_p,
        seed=seed,
        stop_ids=1,
        pin_emit=pin_emit,
        replay_storage=replay_storage,
        schedule_stats=stats,
        paged_cache=paged_cache,
        pad_decode_width=pad_decode_width,
    )


def test_continuous_refill_reuses_freed_slots_without_censoring():
    model = _FakeContinuousModel()
    stats = ContinuousScheduleStats()
    results = _run(
        model,
        [_chunk(1), _chunk(3), _chunk(2)],
        stats=stats,
    )

    assert model.bank_builds == 1
    assert model.admission_calls == [[[0, 1], [2, 3]], [[0, 1]]]
    assert [batch.action_mask.sum(1).tolist() for batch in results] == [
        [1, 1],
        [3, 3],
        [2, 2],
    ]
    # Physical lanes 0/1 restart at the prompt seam when origin 2 refills
    # them; lanes 2/3 retain their independent logical positions.
    assert model.step_calls[:3] == [
        ([0, 1, 2, 3], [3, 3, 3, 3]),
        ([0, 1, 2, 3], [3, 3, 4, 4]),
        ([0, 1, 2, 3], [4, 4, 5, 5]),
    ]
    assert stats.admitted_groups == 3
    assert stats.admitted_rows == stats.evicted_rows == 6
    assert stats.active_rows_max == 4
    assert stats.metrics()["decode_slot_occupancy"] == 1.0


def test_continuous_sampler_accepts_one_uniform_per_request():
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    sampled = _top_p_from_uniform(logits, torch.tensor([0.2]), 1.0, 1.0)
    assert sampled.item() == 0


def test_padded_decode_width_changes_nothing_a_live_row_can_observe():
    """Padding rows are for the compiler's batch bucket, not for the model.

    This runs on CPU only because ``pad_decode_width`` is injectable; the
    production gate is ``device.type == "cuda"``, so without the override the
    only code that steps rows outside the logical batch would have no test at
    all. That code used to reserve a currently-free LANE per padding row,
    which is safe only while a slot permanently owns one -- precisely the
    assumption the page table exists to remove.
    """
    # Three-row groups in a four-row pool: the batch never lands on a bucket
    # boundary on its own, so every step is padded.
    chunks = [_chunk(1), _chunk(3), _chunk(2)]
    plain = _run(_FakeContinuousModel(), chunks, repeats=3)
    padded_model = _FakeContinuousModel()
    padded = _run(padded_model, chunks, repeats=3, pad_decode_width=True)

    assert padded_model.padded_widths, "no step was actually padded"
    assert max(padded_model.padded_widths) == 4
    for left, right in zip(plain, padded, strict=True):
        torch.testing.assert_close(left.action_mask, right.action_mask)
        torch.testing.assert_close(left.token_ids, right.token_ids)


def test_full_capacity_completion_refills_on_the_next_iteration():
    model = _FakeContinuousModel()
    results = _run(
        model,
        [_chunk(1), _chunk(2)],
        repeats=2,
        capacity=2,
    )

    assert model.admission_calls == [[[0, 1]], [[0, 1]]]
    assert model.step_calls[:2] == [
        ([0, 1], [3, 3]),
        ([0, 1], [3, 3]),
    ]
    assert [batch.action_mask.sum(1).tolist() for batch in results] == [
        [1, 1],
        [2, 2],
    ]


def test_scheduler_reuses_a_caller_owned_paged_cache():
    model = _FakeContinuousModel()
    chunks = [_chunk(1), _chunk(2)]
    cache = model.make_paged_generation_cache(
        4, 3 + 5, torch.device("cpu")
    )
    assert model.cache_allocations == 1

    results = _run(model, chunks, paged_cache=cache)

    assert model.cache_allocations == 1
    assert sum(batch.kind.size(0) for batch in results) == 4


def test_admission_never_splits_a_prompt_group():
    model = _FakeContinuousModel()
    stats = ContinuousScheduleStats()
    results = _run(
        model,
        [_chunk(1), _chunk(1), _chunk(1)],
        repeats=2,
        capacity=3,
        stats=stats,
    )

    assert model.bank_builds == 1
    assert model.admission_calls == [[[0, 1]], [[0, 1]], [[0, 1]]]
    assert stats.active_rows_max == 2
    assert stats.free_rows_min == 1
    assert [batch.action_mask.sum(1).tolist() for batch in results] == [
        [1, 1],
        [1, 1],
        [1, 1],
    ]


def test_request_rng_is_stable_across_capacity_and_refill_order():
    chunks = [_chunk(9), _chunk(9), _chunk(9), _chunk(9)]
    narrow = _run(
        _FakeContinuousModel(stochastic_tokens=True),
        chunks,
        repeats=1,
        capacity=2,
        max_new_tokens=4,
        seed=1234,
    )
    wide = _run(
        _FakeContinuousModel(stochastic_tokens=True),
        chunks,
        repeats=1,
        capacity=4,
        max_new_tokens=4,
        seed=1234,
    )

    for narrow_batch, wide_batch in zip(narrow, wide, strict=True):
        torch.testing.assert_close(
            narrow_batch.token_ids, wide_batch.token_ids, rtol=0, atol=0
        )
        torch.testing.assert_close(
            narrow_batch.action_mask, wide_batch.action_mask, rtol=0, atol=0
        )


def test_request_rng_is_stable_across_chunk_partitioning():
    flat_prompts, flat_lengths = _chunk(9, 9, 9, 9)
    paired_chunks = [
        (flat_prompts[:2], flat_lengths[:2]),
        (flat_prompts[2:], flat_lengths[2:]),
    ]
    singleton_chunks = [
        (flat_prompts[index : index + 1], flat_lengths[index : index + 1])
        for index in range(4)
    ]
    paired = _run(
        _FakeContinuousModel(stochastic_tokens=True),
        paired_chunks,
        repeats=2,
        capacity=4,
        max_new_tokens=4,
        seed=1234,
    )
    singleton = _run(
        _FakeContinuousModel(stochastic_tokens=True),
        singleton_chunks,
        repeats=2,
        capacity=2,
        max_new_tokens=4,
        seed=1234,
    )
    def generated(rows):
        result = []
        for batch in rows:
            for row in range(batch.token_ids.size(0)):
                count = int(batch.action_mask[row].sum())
                result.append(
                    batch.token_ids[
                        row, batch.prompt_length : batch.prompt_length + count
                    ].tolist()
                )
        return result

    assert generated(paired) == generated(singleton)


def test_request_rng_changes_with_pool_seed():
    chunks = [_chunk(9), _chunk(9)]
    first = _run(
        _FakeContinuousModel(stochastic_tokens=True),
        chunks,
        repeats=2,
        capacity=4,
        max_new_tokens=5,
        seed=7,
    )
    second = _run(
        _FakeContinuousModel(stochastic_tokens=True),
        chunks,
        repeats=2,
        capacity=4,
        max_new_tokens=5,
        seed=8,
    )

    assert any(
        not left.token_ids.equal(right.token_ids)
        for left, right in zip(first, second, strict=True)
    )


def test_nucleus_sampling_remains_request_stable():
    chunks = [_chunk(9), _chunk(9), _chunk(9)]
    narrow = _run(
        _FakeContinuousModel(stochastic_tokens=True),
        chunks,
        repeats=1,
        capacity=1,
        max_new_tokens=3,
        seed=99,
        top_p=0.7,
    )
    wide = _run(
        _FakeContinuousModel(stochastic_tokens=True),
        chunks,
        repeats=1,
        capacity=3,
        max_new_tokens=3,
        seed=99,
        top_p=0.7,
    )
    for left, right in zip(narrow, wide, strict=True):
        torch.testing.assert_close(
            left.token_ids, right.token_ids, rtol=0, atol=0
        )


def test_latent_rng_is_stable_without_replay_hidden_storage():
    chunks = [_chunk(1), _chunk(1), _chunk(1)]
    narrow = _run(
        _FakeContinuousModel(),
        chunks,
        repeats=1,
        capacity=1,
        max_new_tokens=2,
        seed=71,
        pin_emit=False,
        replay_storage=False,
    )
    wide = _run(
        _FakeContinuousModel(),
        chunks,
        repeats=1,
        capacity=3,
        max_new_tokens=2,
        seed=71,
        pin_emit=False,
        replay_storage=False,
    )
    for left, right in zip(narrow, wide, strict=True):
        assert left.hiddens.size(-1) == 0
        torch.testing.assert_close(
            left.token_ids, right.token_ids, rtol=0, atol=0
        )
        torch.testing.assert_close(
            left.action_mask, right.action_mask, rtol=0, atol=0
        )


def test_left_padding_start_is_carried_across_refill_steps():
    model = _FakeContinuousModel()
    prompts, lengths = _chunk(2)
    prompts[0, 0] = 0
    lengths[0] = 2
    _run(model, [(prompts, lengths)], repeats=2, capacity=2)

    assert model.step_kv_starts
    assert all(starts == [1, 1] for starts in model.step_kv_starts)


def test_real_paged_model_runs_refill_with_independent_positions():
    model = LatentThoughtModel(_backbone()).eval()
    first_prompts = torch.tensor([[0, 0, 7, 11], [0, 5, 9, 13]])
    second_prompts = torch.tensor([[3, 17, 19, 23]])
    stats = ContinuousScheduleStats()
    results = rollout_continuous_refill_groups(
        model,
        [first_prompts, second_prompts],
        [torch.tensor([2, 3]), torch.tensor([4])],
        prompt_repeats=2,
        capacity_rows=4,
        max_new_tokens=2,
        max_stream_steps=2,
        temperature=1.0,
        top_p=1.0,
        seed=101,
        cache_dtype=torch.float32,
        pin_emit=True,
        schedule_stats=stats,
    )

    assert [batch.action_mask.sum(1).tolist() for batch in results] == [
        [2, 2, 2, 2],
        [2, 2],
    ]
    assert all(torch.isfinite(batch.token_ids).all() for batch in results)
    assert stats.admitted_groups == 3
    assert stats.evicted_rows == 6


def _assert_split_groups_replay_carries(model, results, chunk_lengths, repeats):
    """Split each chunk batch per group, then check carries against replay.

    Replay parity holds for the batches the trainer actually replays: the
    per-group, pad-trimmed splits from ``split_rollout_groups``. The raw
    per-chunk batch is an intermediate — its left-padded rows replay
    differently on any trunk with live projections, because the rollout
    excludes pad positions structurally (kv_starts / prefill masking) while
    the parallel replay only zeroes their inputs, and residual biases
    re-inflate them from the first block on. The trainer never replays a
    chunk batch, so neither do these tests.
    """
    checked = 0
    for batch, lengths in zip(results, chunk_lengths, strict=True):
        expanded = lengths.repeat_interleave(repeats)
        for group in split_rollout_groups(batch, repeats, expanded):
            carried = generated_slot_mask(group)
            assert bool(carried.any())
            assert group.hiddens.size(-1) == model.backbone.tok_emb.embedding_dim
            assert float(group.hiddens[~carried].abs().sum()) == 0.0
            with torch.no_grad():
                _, beliefs = replay_beliefs(model, group)
            tail = carried[:, 1:]
            torch.testing.assert_close(
                group.hiddens[:, 1:][tail],
                beliefs[:, :-1][tail],
                rtol=1e-4,
                atol=1e-5,
            )
            checked += 1
    assert checked == sum(lengths.numel() for lengths in chunk_lengths)


def test_real_paged_model_hidden_carry_matches_dense_replay():
    """Scheduler-stored carries must equal the dense replay reconstruction.

    The paged decode loop and ``replay_beliefs`` are independent code paths;
    the stored hidden at slot t is the belief that emitted token t, so it
    must match the replayed belief at t-1 wherever the carry flag is set. A
    live combiner makes the carried content feed back into later beliefs,
    so a corrupted store shows up as a cascading mismatch, not a no-op. The
    trunk projections are livened too: the fresh init's zeroed outputs make
    the blocks an identity residual stream, which would vacuously hide any
    disagreement between the paged rollout and the replay.
    """
    model = LatentThoughtModel(_backbone()).eval()
    with torch.no_grad():
        for block in model.backbone.blocks:
            block.attn.proj.weight.normal_(std=0.02)
            block.mlp.proj.weight.normal_(std=0.02)
        model.combiner.carry.weight.normal_(std=0.02)
        model.combiner.type_bias.normal_(std=0.02)
        for mlp in model.combiner.mlps:
            mlp.proj.weight.normal_(std=0.02)
    first_prompts = torch.tensor([[0, 0, 7, 11], [0, 5, 9, 13]])
    second_prompts = torch.tensor([[3, 17, 19, 23]])
    chunk_lengths = [torch.tensor([2, 3]), torch.tensor([4])]
    results = rollout_continuous_refill_groups(
        model,
        [first_prompts, second_prompts],
        chunk_lengths,
        prompt_repeats=2,
        capacity_rows=4,
        max_new_tokens=3,
        max_stream_steps=3,
        temperature=1.0,
        top_p=1.0,
        seed=101,
        cache_dtype=torch.float32,
        pin_emit=False,
    )

    assert results
    _assert_split_groups_replay_carries(model, results, chunk_lengths, 2)


def test_real_kda_model_runs_refill_with_independent_positions():
    """The recurrent hybrid trunk goes through the same scheduler lifecycle.

    Same shape assertions as the dense variant: admission, ragged eviction,
    and refill must not depend on the cache being KV-addressed. The KDA
    lanes ride the 4-tuple arenas in the same ``PagedGenerationCache``.
    """
    model = LatentThoughtModel(_kda_backbone()).eval()
    first_prompts = torch.tensor([[0, 0, 7, 11], [0, 5, 9, 13]])
    second_prompts = torch.tensor([[3, 17, 19, 23]])
    stats = ContinuousScheduleStats()
    results = rollout_continuous_refill_groups(
        model,
        [first_prompts, second_prompts],
        [torch.tensor([2, 3]), torch.tensor([4])],
        prompt_repeats=2,
        capacity_rows=4,
        max_new_tokens=2,
        max_stream_steps=2,
        temperature=1.0,
        top_p=1.0,
        seed=101,
        cache_dtype=torch.float32,
        pin_emit=True,
        schedule_stats=stats,
    )

    assert [batch.action_mask.sum(1).tolist() for batch in results] == [
        [2, 2, 2, 2],
        [2, 2],
    ]
    assert all(torch.isfinite(batch.token_ids).all() for batch in results)
    assert stats.admitted_groups == 3
    assert stats.evicted_rows == 6


def test_real_kda_model_hidden_carry_matches_dense_replay():
    """Paged recurrent decode must agree with the full-sequence replay.

    The strongest end-to-end statement for the KDA port: the scheduler's
    decode loop advances lane-gathered conv windows and delta-rule state one
    token at a time, while ``replay_beliefs`` re-prices the same stream
    through the parallel reference recurrence. A live combiner feeds stored
    carries back into later beliefs, so any lane-state corruption cascades
    into a mismatch instead of cancelling out.
    """
    model = LatentThoughtModel(_kda_backbone()).eval()
    with torch.no_grad():
        model.combiner.carry.weight.normal_(std=0.02)
        model.combiner.type_bias.normal_(std=0.02)
        for mlp in model.combiner.mlps:
            mlp.proj.weight.normal_(std=0.02)
    first_prompts = torch.tensor([[0, 0, 7, 11], [0, 5, 9, 13]])
    second_prompts = torch.tensor([[3, 17, 19, 23]])
    chunk_lengths = [torch.tensor([2, 3]), torch.tensor([4])]
    results = rollout_continuous_refill_groups(
        model,
        [first_prompts, second_prompts],
        chunk_lengths,
        prompt_repeats=2,
        capacity_rows=4,
        max_new_tokens=3,
        max_stream_steps=3,
        temperature=1.0,
        top_p=1.0,
        seed=101,
        cache_dtype=torch.float32,
        pin_emit=False,
    )

    assert results
    _assert_split_groups_replay_carries(model, results, chunk_lengths, 2)


def test_kda_padded_decode_width_changes_nothing_a_live_row_can_observe():
    """The recurrent twin of the fake-model padding test.

    The fake model never runs the arena write path, so on a KDA trunk the
    only thing standing between a padding row and a live lane's delta-rule
    state is the scratch-lane redirect. A three-row group in a four-row
    pool pads every decode step; forcing ``pad_decode_width`` on and off
    must leave tokens and logprobs bit-identical. The stored carries get a
    tight fp32 closeness bound instead: pad off runs the decode GEMMs at
    width 3 and pad on at width 4, and BLAS kernel selection alone moves
    beliefs by ~7e-7 across widths — while a broken scratch redirect moves
    them by ~0.3 (the measured lane-corruption scale). The combiner is
    livened so a corrupted carry would compound into later tokens instead
    of cancelling out.
    """
    assert _decode_execution_width(3, 4, pending_groups=False) == 4
    model = LatentThoughtModel(_kda_backbone()).eval()
    with torch.no_grad():
        model.combiner.carry.weight.normal_(std=0.02)
        model.combiner.type_bias.normal_(std=0.02)
        for mlp in model.combiner.mlps:
            mlp.proj.weight.normal_(std=0.02)
    prompts = torch.tensor([[0, 7, 11, 13], [5, 9, 13, 17], [3, 17, 19, 23]])
    lengths = torch.tensor([3, 4, 4])

    def run(pad: bool):
        return rollout_continuous_refill_groups(
            model,
            [prompts],
            [lengths],
            prompt_repeats=1,
            capacity_rows=4,
            max_new_tokens=4,
            max_stream_steps=4,
            temperature=1.0,
            top_p=1.0,
            seed=101,
            cache_dtype=torch.float32,
            pin_emit=False,
            pad_decode_width=pad,
        )

    plain = run(False)
    padded = run(True)
    for left, right in zip(plain, padded, strict=True):
        assert torch.equal(left.token_ids, right.token_ids)
        assert torch.equal(left.old_token_logprobs, right.old_token_logprobs)
        torch.testing.assert_close(
            left.hiddens, right.hiddens, rtol=1e-5, atol=1e-5
        )


def test_scheduler_rejects_capacity_smaller_than_one_group():
    model = _FakeContinuousModel()
    chunk = _chunk(1)
    try:
        _run(model, [chunk], repeats=2, capacity=1)
    except ValueError as error:
        assert "whole prompt group" in str(error)
    else:
        raise AssertionError("scheduler accepted a split group capacity")


def test_warmup_decode_width_buckets_covers_and_preserves() -> None:
    """Warmup must visit every width the scheduler can request, and a
    warmed arena must produce bit-identical rollouts to a fresh one —
    dead-row warmup writes may only touch pages no slot owns."""
    capacity = 4
    first_prompts = torch.tensor([[0, 0, 7, 11], [0, 5, 9, 13]])
    second_prompts = torch.tensor([[3, 17, 19, 23]])

    def run(cache):
        return rollout_continuous_refill_groups(
            LatentThoughtModel(_backbone()).eval(),
            [first_prompts, second_prompts],
            [torch.tensor([2, 3]), torch.tensor([4])],
            prompt_repeats=2,
            capacity_rows=capacity,
            max_new_tokens=2,
            max_stream_steps=2,
            temperature=1.0,
            top_p=1.0,
            seed=101,
            cache_dtype=torch.float32,
            pin_emit=True,
            paged_cache=cache,
        )

    model = LatentThoughtModel(_backbone()).eval()
    cache = model.make_paged_generation_cache(
        capacity, 4 + 2, torch.device("cpu"), dtype=torch.float32
    )
    widths = warmup_decode_width_buckets(model, cache, capacity)
    assert widths == sorted(widths, reverse=True)
    assert widths[0] == capacity
    for active in range(1, capacity + 1):
        for pending in (False, True):
            assert (
                _decode_execution_width(
                    active, capacity, pending_groups=pending
                )
                in widths
            )

    warmed = run(cache)
    fresh_model = LatentThoughtModel(_backbone()).eval()
    fresh = run(
        fresh_model.make_paged_generation_cache(
            capacity, 4 + 2, torch.device("cpu"), dtype=torch.float32
        )
    )
    for warmed_batch, fresh_batch in zip(warmed, fresh, strict=True):
        assert torch.equal(warmed_batch.token_ids, fresh_batch.token_ids)
        assert torch.equal(
            warmed_batch.old_token_logprobs, fresh_batch.old_token_logprobs
        )
        assert torch.equal(warmed_batch.hiddens, fresh_batch.hiddens)


def test_warmup_decode_width_buckets_preserves_kda_arenas() -> None:
    """Warmup on a recurrent trunk writes state, not just KV.

    Every warmup row is dead, so on a KDA cache each pass exercises the
    scratch-lane redirect at every width bucket — three unconditional
    arena writes per recurrent layer per step. A warmed cache must still
    produce bit-identical rollouts to a fresh one; anything else means a
    warmup write landed inside ``[0, capacity)``.
    """
    capacity = 4
    model = LatentThoughtModel(_kda_backbone()).eval()
    first_prompts = torch.tensor([[0, 0, 7, 11], [0, 5, 9, 13]])
    second_prompts = torch.tensor([[3, 17, 19, 23]])

    def run(cache):
        return rollout_continuous_refill_groups(
            model,
            [first_prompts, second_prompts],
            [torch.tensor([2, 3]), torch.tensor([4])],
            prompt_repeats=2,
            capacity_rows=capacity,
            max_new_tokens=2,
            max_stream_steps=2,
            temperature=1.0,
            top_p=1.0,
            seed=101,
            cache_dtype=torch.float32,
            pin_emit=True,
            paged_cache=cache,
        )

    cache = model.make_paged_generation_cache(
        capacity, 4 + 2, torch.device("cpu"), dtype=torch.float32
    )
    widths = warmup_decode_width_buckets(model, cache, capacity)
    assert widths and widths[0] == capacity
    warmed = run(cache)
    fresh = run(
        model.make_paged_generation_cache(
            capacity, 4 + 2, torch.device("cpu"), dtype=torch.float32
        )
    )
    for warmed_batch, fresh_batch in zip(warmed, fresh, strict=True):
        assert torch.equal(warmed_batch.token_ids, fresh_batch.token_ids)
        assert torch.equal(
            warmed_batch.old_token_logprobs, fresh_batch.old_token_logprobs
        )
        assert torch.equal(warmed_batch.hiddens, fresh_batch.hiddens)
