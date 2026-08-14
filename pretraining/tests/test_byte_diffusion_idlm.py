"""CPU correctness tests for the reference I-DLM primitives."""

from __future__ import annotations

import pytest
import torch

from pretraining.byte_diffusion.idlm import (
    IGNORE_INDEX,
    auto_balanced_loss,
    categorical_from_uniform,
    isd_acceptance_probability,
    isd_commit_prefix,
    idlm_loss,
    make_training_layout,
    positive_residual_distribution,
    shifted_targets,
    strict_attention_mask,
)


def test_training_layout_is_all_mask_then_clean_without_target_leakage() -> None:
    clean = torch.tensor([[10, 11, 12, 13, 77]])
    valid = torch.tensor([[True, True, True, True, False]])
    layout = make_training_layout(
        clean,
        valid,
        mask_id=90,
        pad_id=91,
        block_size=2,
    )
    assert layout.proposal_ids.tolist() == [[90, 90, 90, 90, 91]]
    assert layout.input_ids.tolist() == [
        [90, 90, 90, 90, 91, 10, 11, 12, 13, 77]
    ]


def test_strict_attention_topology_exhaustively_matches_paper_rule() -> None:
    length = 6
    block = 2
    clean = torch.arange(length)[None]
    valid = torch.ones_like(clean, dtype=torch.bool)
    layout = make_training_layout(
        clean,
        valid,
        mask_id=90,
        pad_id=91,
        block_size=block,
    )
    allowed = strict_attention_mask(layout)[0]

    for query in range(2 * length):
        for key in range(2 * length):
            if query < length:
                query_position = query
                if key < length:
                    key_position = key
                    expected = (
                        key_position // block == query_position // block
                        and key_position <= query_position
                    )
                else:
                    key_position = key - length
                    expected = key_position // block < query_position // block
            else:
                query_position = query - length
                expected = key >= length and key - length <= query_position
            assert bool(allowed[query, key]) is expected


def test_clean_attention_is_exact_causal_parity_and_never_reads_proposals() -> None:
    clean = torch.arange(5)[None]
    valid = torch.tensor([[True, True, True, True, False]])
    layout = make_training_layout(
        clean,
        valid,
        mask_id=90,
        pad_id=91,
        block_size=3,
    )
    allowed = strict_attention_mask(layout)
    length = layout.sequence_length
    positions = torch.arange(length)
    expected = (
        (positions[None, :] <= positions[:, None])[None]
        & valid[:, :, None]
        & valid[:, None, :]
    )
    assert torch.equal(allowed[:, length:, length:], expected)
    assert not bool(allowed[:, length:, :length].any())


def test_segment_ids_prevent_attention_and_targets_crossing_documents() -> None:
    clean = torch.tensor([[10, 11, 91, 20, 21, 22]])
    valid = torch.tensor([[True, True, False, True, True, True]])
    segments = torch.tensor([[0, 0, -1, 1, 1, 1]])
    layout = make_training_layout(
        clean,
        valid,
        mask_id=90,
        pad_id=91,
        block_size=2,
        segment_ids=segments,
    )
    allowed = strict_attention_mask(layout)[0]
    length = layout.sequence_length
    assert not bool(allowed[3, length : length + 2].any())
    assert not bool(allowed[length + 3, length : length + 2].any())

    targets = shifted_targets(layout)
    expected = torch.tensor(
        [[11, IGNORE_INDEX, IGNORE_INDEX, 21, 22, IGNORE_INDEX]]
    )
    assert torch.equal(targets.proposal, expected)
    assert torch.equal(targets.clean, expected)


def test_packed_document_positions_reset_off_physical_block_phase() -> None:
    clean = torch.tensor([[10, 11, 12, 20, 21, 22, 23]])
    valid = torch.ones_like(clean, dtype=torch.bool)
    segments = torch.tensor([[0, 0, 0, 1, 1, 1, 1]])
    layout = make_training_layout(
        clean,
        valid,
        mask_id=90,
        pad_id=91,
        block_size=2,
        segment_ids=segments,
    )
    assert layout.positions.tolist() == [[0, 1, 2, 0, 1, 2, 3]]

    allowed = strict_attention_mask(layout)[0]
    length = layout.sequence_length
    # Document 2 begins at physical column 3, halfway through a global
    # two-column block. Relative positions 0 and 1 must nevertheless share
    # its first proposal block and must not read any clean token yet.
    assert allowed[4, 3:5].tolist() == [True, True]
    assert not bool(allowed[4, length + 3 : length + 5].any())
    # Relative position 2 starts document 2's next block: it reads the first
    # clean block and only its own causal proposal position.
    assert allowed[5, 3:6].tolist() == [False, False, True]
    assert allowed[5, length + 3 : length + 6].tolist() == [True, True, False]
    # The clean half remains document-local causal attention.
    assert allowed[length + 4, length :].tolist() == [
        False,
        False,
        False,
        True,
        True,
        False,
        False,
    ]


def test_layout_rejects_positions_that_do_not_reset_per_document() -> None:
    clean = torch.tensor([[10, 11, 12, 20, 21]])
    valid = torch.ones_like(clean, dtype=torch.bool)
    segments = torch.tensor([[0, 0, 0, 1, 1]])
    with pytest.raises(ValueError, match="start at zero"):
        make_training_layout(
            clean,
            valid,
            mask_id=90,
            pad_id=91,
            block_size=2,
            segment_ids=segments,
            positions=torch.tensor([[0, 1, 2, 1, 2]]),
        )


def test_shifted_labels_apply_to_both_paths_and_do_not_reset_at_blocks() -> None:
    clean = torch.tensor([[40, 41, 42, 43, 44]])
    valid = torch.ones_like(clean, dtype=torch.bool)
    layout = make_training_layout(
        clean,
        valid,
        mask_id=90,
        pad_id=91,
        block_size=2,
    )
    targets = shifted_targets(layout)
    expected = torch.tensor([[41, 42, 43, 44, IGNORE_INDEX]])
    assert torch.equal(targets.proposal, expected)
    assert torch.equal(targets.clean, expected)
    assert torch.equal(targets.combined, torch.cat((expected, expected), dim=1))


def test_auto_balance_detaches_ratio_without_scalar_device_conversion() -> None:
    proposal = torch.tensor(2.0, requires_grad=True)
    clean = torch.tensor(4.0, requires_grad=True)
    total, scale = auto_balanced_loss(proposal, clean)
    torch.testing.assert_close(scale, torch.tensor(0.5))
    torch.testing.assert_close(total, torch.tensor(4.0))
    assert not scale.requires_grad
    total.backward()
    torch.testing.assert_close(proposal.grad, torch.tensor(1.0))
    torch.testing.assert_close(clean.grad, torch.tensor(0.5))


def test_auto_balance_is_fullgraph_compilable() -> None:
    compiled = torch.compile(auto_balanced_loss, backend="eager", fullgraph=True)
    total, scale = compiled(torch.tensor(3.0), torch.tensor(2.0))
    torch.testing.assert_close(total, torch.tensor(6.0))
    torch.testing.assert_close(scale, torch.tensor(1.5))


def test_dense_idlm_loss_uses_both_shifted_pathways_and_target_counts() -> None:
    clean = torch.tensor([[0, 0, 0]])
    valid = torch.ones_like(clean, dtype=torch.bool)
    layout = make_training_layout(
        clean,
        valid,
        mask_id=2,
        pad_id=3,
        block_size=2,
    )
    targets = shifted_targets(layout, ignore_index=-7)

    def logits_for_nll(nll: float) -> torch.Tensor:
        probability = torch.exp(torch.tensor(-nll, dtype=torch.float64))
        row = torch.stack((probability.log(), (1.0 - probability).log()))
        return row.expand(1, 3, 2).clone()

    objective = idlm_loss(logits_for_nll(2.0), logits_for_nll(4.0), targets)
    torch.testing.assert_close(objective.proposal, torch.tensor(2.0, dtype=torch.float64))
    torch.testing.assert_close(objective.clean, torch.tensor(4.0, dtype=torch.float64))
    torch.testing.assert_close(objective.clean_scale, torch.tensor(0.5, dtype=torch.float64))
    torch.testing.assert_close(objective.total, torch.tensor(4.0, dtype=torch.float64))
    assert objective.proposal_targets.item() == 2
    assert objective.clean_targets.item() == 2


def test_exact_pq_correction_has_anchor_distribution() -> None:
    p = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    q = torch.tensor([0.5, 0.1, 0.4], dtype=torch.float64)
    proposal_ids = torch.arange(3)
    acceptance = isd_acceptance_probability(
        p.expand(3, -1), q.expand(3, -1), proposal_ids
    )
    residual = positive_residual_distribution(p, q)

    accepted_mass = q * acceptance
    rejection_mass = (q * (1.0 - acceptance)).sum()
    corrected_marginal = accepted_mass + rejection_mass * residual
    torch.testing.assert_close(corrected_marginal, p, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "bad",
    [
        torch.tensor([0.5, -0.1, 0.6]),
        torch.tensor([0.5, float("nan"), 0.5]),
        torch.zeros(3),
    ],
)
def test_pq_primitives_reject_invalid_probability_rows(bad: torch.Tensor) -> None:
    with pytest.raises(RuntimeError, match="categorical"):
        positive_residual_distribution(torch.tensor([0.2, 0.5, 0.3]), bad)


def test_vectorized_inverse_cdf_respects_probability_intervals() -> None:
    probabilities = torch.tensor(
        [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]
    )
    uniforms = torch.tensor([0.1, 0.2, 0.999])
    assert categorical_from_uniform(probabilities, uniforms).tolist() == [0, 1, 2]


def test_isd_commit_truncates_after_rejection_and_emits_all_accept_bonus() -> None:
    anchor_ids = torch.tensor([7, 8])
    proposal_ids = torch.tensor([[0, 1, 0], [0, 0, 0]])

    p = torch.tensor(
        [
            [[0.6, 0.4], [0.2, 0.8], [0.7, 0.3]],
            [[0.6, 0.4], [0.1, 0.9], [0.5, 0.5]],
        ]
    )
    q = torch.tensor(
        [
            [[0.6, 0.4], [0.2, 0.8], [0.7, 0.3]],
            [[0.6, 0.4], [0.9, 0.1], [0.5, 0.5]],
        ]
    )
    result = isd_commit_prefix(
        anchor_ids,
        p,
        q,
        proposal_ids,
        accept_uniforms=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]]
        ),
        residual_uniforms=torch.zeros(2, 3),
        bonus_p=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        bonus_uniforms=torch.zeros(2),
    )

    assert result.proposal_accepted.tolist() == [
        [True, True, True],
        [True, False, True],
    ]
    assert result.valid.tolist() == [
        [True, True, True, True, True],
        [True, True, True, False, False],
    ]
    assert result.all_proposals_accepted.tolist() == [True, False]
    assert result.ids[0].tolist() == [7, 0, 1, 0, 1]
    # Row 1 proposal 2 is rejected and corrected to token 1. Proposal 3 and
    # the bonus remain present only as shape-stable, invalid storage.
    assert result.ids[1, :3].tolist() == [8, 0, 1]


def test_stride_one_degenerates_to_anchor_plus_bonus_without_proposals() -> None:
    empty_probabilities = torch.empty(1, 0, 2)
    empty_ids = torch.empty(1, 0, dtype=torch.long)
    result = isd_commit_prefix(
        torch.tensor([7]),
        empty_probabilities,
        empty_probabilities,
        empty_ids,
        accept_uniforms=torch.empty(1, 0),
        residual_uniforms=torch.empty(1, 0),
        bonus_p=torch.tensor([[0.0, 1.0]]),
        bonus_uniforms=torch.tensor([0.0]),
    )
    assert result.ids.tolist() == [[7, 1]]
    assert result.valid.tolist() == [[True, True]]
    assert result.all_proposals_accepted.tolist() == [True]
