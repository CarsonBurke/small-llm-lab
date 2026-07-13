from __future__ import annotations

import torch

from postraining.adaptation_core import (
    PAD_KIND,
    THOUGHT_KIND,
    TOKEN_KIND,
    assemble_stream_inputs,
    build_thought_plan,
    copy_last_latent_mse,
    gather_stream_targets,
    weighted_cross_entropy,
)


def test_no_inserts_is_the_identity_stream():
    mask = torch.zeros(2, 5, dtype=torch.bool)
    plan = build_thought_plan(mask)
    assert plan.stream_length == 5
    assert torch.all(plan.kind == TOKEN_KIND)
    assert torch.equal(plan.source, torch.arange(5).expand(2, 5))
    latents = torch.randn(2, 5, 3)
    thoughts = torch.randn(2, 5, 3)
    torch.testing.assert_close(assemble_stream_inputs(plan, latents, thoughts), latents)


def test_interleaving_layout_and_padding():
    mask = torch.tensor([[False, True, False, False], [False, False, False, False]])
    plan = build_thought_plan(mask, thought_ce_weight=0.5)
    assert plan.stream_length == 5
    assert plan.kind[0].tolist() == [TOKEN_KIND, TOKEN_KIND, THOUGHT_KIND, TOKEN_KIND, TOKEN_KIND]
    assert plan.source[0].tolist() == [0, 1, 1, 2, 3]
    assert plan.kind[1].tolist() == [TOKEN_KIND] * 4 + [PAD_KIND]
    assert plan.ce_weight[0].tolist() == [1.0, 1.0, 0.5, 1.0, 1.0]
    assert plan.ce_weight[1].tolist() == [1.0] * 4 + [0.0]
    assert plan.latent_weight[1].tolist() == [1.0] * 4 + [0.0]


def test_thought_slot_consumes_thought_and_targets_the_upcoming_token():
    mask = torch.tensor([[False, True, False]])
    plan = build_thought_plan(mask)
    latents = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)
    thoughts = -torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)
    stream = assemble_stream_inputs(plan, latents, thoughts)
    torch.testing.assert_close(stream[0, 2], thoughts[0, 1])
    torch.testing.assert_close(stream[0, 1], latents[0, 1])
    targets = torch.tensor([[7, 8, 9]])
    stream_targets = gather_stream_targets(plan, targets)
    # thought after token 1 predicts token 2 — the same target as token 1.
    assert stream_targets[0].tolist() == [7, 8, 8, 9]


def test_multiple_inserts_keep_pads_strictly_at_the_tail():
    torch.manual_seed(3)
    mask = torch.rand(8, 33) < 0.3
    plan = build_thought_plan(mask)
    for row in range(8):
        kinds = plan.kind[row].tolist()
        if PAD_KIND in kinds:
            first_pad = kinds.index(PAD_KIND)
            assert all(kind == PAD_KIND for kind in kinds[first_pad:])
        tokens = [source for kind, source in zip(kinds, plan.source[row].tolist()) if kind == TOKEN_KIND]
        assert tokens == list(range(33))


def test_weighted_cross_entropy_matches_manual_reduction():
    torch.manual_seed(5)
    logits = torch.randn(2, 4, 7)
    targets = torch.randint(0, 7, (2, 4))
    weights = torch.tensor([[1.0, 0.5, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0]])
    actual = weighted_cross_entropy(logits, targets, weights)
    manual = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), targets.flatten(), reduction="none"
    )
    expected = (manual * weights.flatten()).sum() / weights.sum()
    torch.testing.assert_close(actual, expected)


def test_full_mask_doubles_the_stream_with_alternating_kinds():
    mask = torch.ones(2, 4, dtype=torch.bool)
    plan = build_thought_plan(mask)
    assert plan.stream_length == 8
    assert plan.kind[0].tolist() == [TOKEN_KIND, THOUGHT_KIND] * 4
    assert plan.source[0].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_insert_at_the_final_position_targets_the_last_real_token():
    mask = torch.tensor([[False, False, True]])
    plan = build_thought_plan(mask)
    assert plan.kind[0].tolist() == [TOKEN_KIND, TOKEN_KIND, TOKEN_KIND, THOUGHT_KIND]
    targets = gather_stream_targets(plan, torch.tensor([[7, 8, 9]]))
    assert targets[0].tolist() == [7, 8, 9, 9]


def test_thought_latent_weight_lands_on_thought_slots_only():
    mask = torch.tensor([[False, True, False]])
    plan = build_thought_plan(mask, thought_ce_weight=0.25, thought_latent_weight=0.5)
    assert plan.ce_weight[0].tolist() == [1.0, 1.0, 0.25, 1.0]
    assert plan.latent_weight[0].tolist() == [1.0, 1.0, 0.5, 1.0]


def test_copy_last_baseline_is_zero_for_static_latents():
    latents = torch.randn(2, 5, 3)
    assert float(copy_last_latent_mse(latents, latents)) == 0.0
    assert float(copy_last_latent_mse(latents, latents + 1.0)) > 0.0
