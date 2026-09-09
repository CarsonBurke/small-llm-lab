from __future__ import annotations

from types import SimpleNamespace

import torch

from postraining.fast_inference import (
    CapturedTrainingRolloutEngine,
    PromptPrefixBank,
    _CompactStaticLayer,
)
from postraining.uno_speculative import (
    UnoTrainingRolloutEngine,
    _UnoStaticLayer,
    commit_cycle_,
    couple_proposals,
    sparse_sampling_support,
    support_probability,
)


def test_support_matches_production_topk_then_nucleus() -> None:
    logits = torch.tensor([[0.4, 0.3, 0.2, 0.1]]).log()
    ids, probabilities = sparse_sampling_support(
        logits,
        temperature=1.0,
        top_k=3,
        top_p=0.6,
    )
    torch.testing.assert_close(ids, torch.tensor([[0, 1, 2]]))
    torch.testing.assert_close(probabilities, torch.tensor([[4 / 7, 3 / 7, 0.0]]))
    found = support_probability(ids, probabilities, torch.tensor([[3, 1, 0, 7]]))
    torch.testing.assert_close(found, torch.tensor([[0.0, 3 / 7, 4 / 7, 0.0]]))


def test_sparse_acceptance_and_residual_reconstruct_target_law() -> None:
    # Exact finite quadrature, not a stochastic tolerance test. Proposal 0 has
    # mass 3/4 and accepts with probability 1/3; rejection corrects to token 1.
    rows = 48
    q_ids = torch.tensor([[[0, 1]]]).expand(rows, -1, -1)
    q = torch.tensor([[[0.75, 0.25]]]).expand(rows, -1, -1)
    p_ids = torch.tensor([[[0, 1], [0, 1]]]).expand(rows, -1, -1)
    p = torch.tensor([[[0.25, 0.75], [0.0, 1.0]]]).expand(rows, -1, -1)
    proposals = torch.cat((torch.zeros(36, 1), torch.ones(12, 1))).long()
    mass = torch.where(proposals == 0, 0.75, 0.25)
    uniforms = torch.cat(
        ((torch.arange(36) + 0.5) / 36, (torch.arange(12) + 0.5) / 12)
    )[:, None]
    prefix, correction = couple_proposals(p_ids, p, q_ids, q, proposals, mass, uniforms)
    output = torch.where(prefix[:, 0].bool(), proposals[:, 0], correction)
    torch.testing.assert_close(
        torch.bincount(output, minlength=2), torch.tensor([12, 36])
    )


def test_disjoint_support_rejects_and_uses_target_only_residual() -> None:
    p_ids = torch.tensor([[[8, 9], [9, 8]]])
    p = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    q_ids = torch.tensor([[[1, 2]]])
    q = torch.tensor([[[1.0, 0.0]]])
    prefix, correction = couple_proposals(
        p_ids,
        p,
        q_ids,
        q,
        torch.tensor([[1]]),
        torch.ones(1, 1),
        torch.tensor([[0.5]]),
    )
    torch.testing.assert_close(prefix, torch.zeros(1, 1, dtype=torch.long))
    torch.testing.assert_close(correction, torch.tensor([8]))


def test_acceptance_stops_at_first_rejection_and_bonus_uses_final_target() -> None:
    q_ids = torch.tensor([[[1, 7], [2, 7], [3, 7]], [[1, 7], [2, 7], [3, 7]]])
    q = torch.tensor([[[1.0, 0.0]]]).expand(2, 3, 2)
    target_ids = torch.tensor(
        [
            [[1, 7], [8, 7], [3, 7], [9, 7]],
            [[1, 7], [2, 7], [3, 7], [9, 7]],
        ]
    )
    p = torch.tensor([[[1.0, 0.0]]]).expand(2, 4, 2)
    prefix, correction = couple_proposals(
        target_ids,
        p,
        q_ids,
        q,
        torch.tensor([[1, 2, 3], [1, 2, 3]]),
        torch.ones(2, 3),
        torch.full((2, 3), 0.5),
    )
    torch.testing.assert_close(prefix, torch.tensor([[1, 0, 0], [1, 1, 1]]))
    torch.testing.assert_close(correction, torch.tensor([8, 9]))


def test_commit_preserves_pending_and_truncates_eos_budget_and_inactive_rows() -> None:
    generated = torch.full((6, 7), -1, dtype=torch.long)
    generated[:, :2] = torch.tensor([40, 41])
    position = torch.full((6,), 2, dtype=torch.long)
    cursor = torch.tensor([5, 4, 3, 2, 1, 0])
    original_cursor = cursor.clone()
    pending = torch.full((6,), 42, dtype=torch.long)
    active = torch.tensor([True, True, True, False, True, True])
    limit = torch.tensor([7, 7, 3, 7, 7, 5])
    free = torch.tensor([10, 99, 10, 10, 10, 10])
    proposals = torch.tensor([[20, 21, 22]]).expand(6, -1).clone()
    proposals[4, 1] = 99
    prefix = torch.tensor(
        [[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    )
    emitted, _ = commit_cycle_(
        generated,
        position,
        cursor,
        pending,
        active,
        limit,
        torch.tensor([99]),
        free,
        proposals,
        prefix,
        torch.full((6,), 90),
    )
    torch.testing.assert_close(emitted, torch.tensor([4, 1, 1, 0, 3, 3]))
    torch.testing.assert_close(cursor, original_cursor + emitted)
    torch.testing.assert_close(pending, torch.tensor([90, 99, 10, 42, 99, 21]))
    torch.testing.assert_close(
        active, torch.tensor([True, False, False, False, False, False])
    )
    assert generated.tolist() == [
        [40, 41, 10, 20, 21, 90, -1],
        [40, 41, 99, -1, -1, -1, -1],
        [40, 41, 10, -1, -1, -1, -1],
        [40, 41, -1, -1, -1, -1, -1],
        [40, 41, 10, 20, 99, -1, -1],
        [40, 41, 10, 20, 21, -1, -1],
    ]


def test_cached_suffix_rollback_never_overwrites_prefix_or_reads_uninitialized_kv() -> (
    None
):
    cursor = torch.tensor([4, 1, 0])
    layer = _UnoStaticLayer(12, cursor, torch.empty(3, 0, dtype=torch.bool))
    draft = torch.arange(12, dtype=torch.float32).reshape(3, 1, 4, 1) + 100
    layer.lazy_initialization(draft, draft)
    layer.key_backing.fill_(float("nan"))
    layer.value_backing.fill_(float("nan"))
    layer.key_backing[0, :4] = 7
    layer.value_backing[0, :4] = 7
    layer.key_backing[1, :1] = 8
    layer.value_backing[1, :1] = 8
    layer.update(draft, draft)
    cursor.add_(1)
    verify = draft + 1000
    layer.update(verify, verify)
    torch.testing.assert_close(layer.key_backing[0, :4], torch.full((4, 1, 1), 7.0))
    torch.testing.assert_close(layer.key_backing[1, :1], torch.full((1, 1, 1), 8.0))
    for row, length in enumerate(cursor.tolist()):
        assert torch.isfinite(layer.key_backing[row, : length + 4]).all()
        torch.testing.assert_close(
            layer.key_backing[row, length - 1, 0, 0], draft[row, 0, 0, 0]
        )
        torch.testing.assert_close(
            layer.key_backing[row, length : length + 4, 0, 0], verify[row, 0, :, 0]
        )
    # Reject immediately: retain only original pending and free-token KV;
    # correction is pending, so the next draft overwrites all rejected suffix.
    cursor.add_(1)
    next_draft = draft + 2000
    layer.update(next_draft, next_draft)
    for row, length in enumerate(cursor.tolist()):
        torch.testing.assert_close(
            layer.key_backing[row, length : length + 4, 0, 0], next_draft[row, 0, :, 0]
        )


def test_single_token_prefill_and_explicit_ar_append_are_distinct() -> None:
    lengths = torch.tensor([1, 1])
    layer = _CompactStaticLayer(4, lengths, torch.ones(2, 1, dtype=torch.bool))
    prompt = torch.tensor([[[[3.0]]], [[[5.0]]]])
    layer.update(prompt, prompt)
    torch.testing.assert_close(layer.key_backing[:, 0, 0, 0], torch.tensor([3.0, 5.0]))
    layer.prefilling = False
    layer.update(prompt + 10, prompt + 10)
    torch.testing.assert_close(
        layer.key_backing[:, :2, 0, 0], torch.tensor([[3.0, 13.0], [5.0, 15.0]])
    )


def test_refill_restores_prompt_last_token_and_excludes_it_from_cursor() -> None:
    engine = object.__new__(UnoTrainingRolloutEngine)
    engine._runtime_device = torch.device("cpu")
    engine.batch_size = 2
    engine.generated = torch.full((2, 8), 77, dtype=torch.long)
    engine.attention_mask = torch.zeros(2, 8, dtype=torch.bool)
    engine.sequence_lengths = torch.tensor([6, 4])
    engine.flash_sequence_lengths = torch.tensor([7, 5], dtype=torch.int32)
    engine.position_ids = torch.tensor([[6], [4]])
    engine.output_position = torch.tensor([3, 2])
    engine.response_limit = torch.tensor([3, 3])
    engine._graph_logits = torch.zeros(2, 2)
    engine.active = torch.tensor([False, True])
    engine._pending = torch.tensor([99, 12])
    engine._prompt_last_tokens = torch.tensor([8, 9])
    layer = _UnoStaticLayer(13, engine.sequence_lengths, engine.attention_mask)
    template = torch.zeros(2, 1, 1, 1)
    layer.lazy_initialization(template, template)
    layer.key_backing.fill_(77)
    layer.value_backing.fill_(77)
    engine.cache = SimpleNamespace(layers=[layer])
    bank = PromptPrefixBank(
        lengths=torch.tensor([2, 3]),
        logits=torch.zeros(2, 2),
        values=torch.empty(0),
        layer_keys=torch.arange(6, dtype=torch.float32).reshape(2, 3, 1, 1, 1),
        layer_values=torch.arange(6, dtype=torch.float32).reshape(2, 3, 1, 1, 1),
    )
    engine._admit_prompt_rows(bank, 1, [0], max_new_tokens=4)
    torch.testing.assert_close(engine.sequence_lengths, torch.tensor([2, 4]))
    torch.testing.assert_close(engine._pending, torch.tensor([9, 12]))
    torch.testing.assert_close(
        layer.key_backing[0, :3, 0, 0], torch.tensor([3.0, 4.0, 5.0])
    )
    torch.testing.assert_close(layer.key_backing[1], torch.full((13, 1, 1), 77.0))
    torch.testing.assert_close(engine.generated[0], torch.zeros(8, dtype=torch.long))
    torch.testing.assert_close(
        engine.generated[1], torch.full((8,), 77, dtype=torch.long)
    )


def test_continuous_scheduler_refills_variable_width_cycles_without_losing_rows() -> (
    None
):
    engine = object.__new__(UnoTrainingRolloutEngine)
    engine.batch_size = 2
    engine.samples_per_prompt = 1
    engine.cache_length = 8
    engine._output_slots_per_cycle = 5
    engine.generated = torch.zeros(2, 8, dtype=torch.long)
    engine._graph_logits = torch.zeros(2, 1)
    engine.cache_position = torch.zeros(1, dtype=torch.long)
    engine.output_position = torch.zeros(2, dtype=torch.long)
    engine.sequence_lengths = torch.zeros(2, dtype=torch.long)
    engine.flash_sequence_lengths = torch.ones(2, dtype=torch.int32)
    engine.position_ids = torch.zeros(2, 1, dtype=torch.long)
    engine.attention_mask = torch.zeros(2, 8, dtype=torch.bool)
    engine.active = torch.zeros(2, dtype=torch.bool)
    engine.response_limit = torch.ones(2, dtype=torch.long)
    engine._pending = torch.zeros(2, dtype=torch.long)
    engine._compile_decode = False
    engine.cache = SimpleNamespace(layers=[])
    engine._synchronize_generation = lambda: None
    engine._ensure_continuous_cache = lambda bank: None
    engine._build_prompt_prefix_bank = lambda *args, **kwargs: None
    admitted = []

    def admit(bank, prompt_index, slots, *, max_new_tokens):
        admitted.append((prompt_index, tuple(slots)))
        for slot in slots:
            engine.generated[slot].zero_()
            engine.output_position[slot] = 0
            engine.sequence_lengths[slot] = 0
            engine.response_limit[slot] = max_new_tokens
            engine._pending[slot] = (98, 9, 19)[prompt_index]
            engine.active[slot] = True

    def cycle():
        free = engine._pending + 1
        proposals = free[:, None] + torch.tensor([1, 2, 3])
        proposals[:, 0] = torch.where(free == 20, 99, proposals[:, 0])
        commit_cycle_(
            engine.generated,
            engine.output_position,
            engine.sequence_lengths,
            engine._pending,
            engine.active,
            engine.response_limit,
            torch.tensor([99]),
            free,
            proposals,
            torch.ones(2, 3, dtype=torch.long),
            free + 4,
        )

    engine._admit_prompt_rows = admit
    engine._continuous_decode_once = cycle
    result = CapturedTrainingRolloutEngine.generate_prompt_pool(
        engine,
        [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])],
        max_new_tokens=6,
        completion_poll_steps=1,
    )
    assert [row.tolist() for row in result.responses] == [
        [99],
        [10, 11, 12, 13, 14, 15],
        [20, 99],
    ]
    assert admitted == [(0, (0,)), (1, (1,)), (2, (0,))]
    assert result.decode_steps == 2
    assert result.useful_tokens == 9
    assert result.capacity_row_steps == 20
