from __future__ import annotations

from types import SimpleNamespace

import torch

from postraining.latent_rollout import generated_slot_mask, replay_beliefs
from postraining.latent_thought import LatentThoughtModel, StepOutput
from postraining.rollout_scheduler import (
    ContinuousScheduleStats,
    _decode_execution_width,
    rollout_continuous_refill_groups,
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


def test_real_paged_model_hidden_carry_matches_dense_replay():
    """Scheduler-stored carries must equal the dense replay reconstruction.

    The paged decode loop and ``replay_beliefs`` are independent code paths;
    the stored hidden at slot t is the belief that emitted token t, so it
    must match the replayed belief at t-1 wherever the carry flag is set. A
    live combiner makes the carried content feed back into later beliefs,
    so a corrupted store shows up as a cascading mismatch, not a no-op.
    """
    model = LatentThoughtModel(_backbone()).eval()
    with torch.no_grad():
        model.combiner.gain.fill_(0.4)
        model.combiner.type_bias.normal_(std=0.02)
        for mlp in model.combiner.mlps:
            mlp.proj.weight.normal_(std=0.02)
    first_prompts = torch.tensor([[0, 0, 7, 11], [0, 5, 9, 13]])
    second_prompts = torch.tensor([[3, 17, 19, 23]])
    results = rollout_continuous_refill_groups(
        model,
        [first_prompts, second_prompts],
        [torch.tensor([2, 3]), torch.tensor([4])],
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
    for batch in results:
        carried = generated_slot_mask(batch)
        assert bool(carried.any())
        assert batch.hiddens.size(-1) == model.backbone.tok_emb.embedding_dim
        assert float(batch.hiddens[~carried].abs().sum()) == 0.0
        with torch.no_grad():
            _, beliefs = replay_beliefs(model, batch)
        tail = carried[:, 1:]
        torch.testing.assert_close(
            batch.hiddens[:, 1:][tail],
            beliefs[:, :-1][tail],
            rtol=1e-4,
            atol=1e-5,
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
