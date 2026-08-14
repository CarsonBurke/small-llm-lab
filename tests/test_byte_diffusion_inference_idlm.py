from __future__ import annotations

from dataclasses import dataclass

import torch
import pytest

from pretraining.byte_diffusion.inference_idlm import (
    generate_idlm_full_replay,
    generate_idlm_fused_replay,
)


@dataclass(frozen=True)
class _Config:
    pad_id: int = 5
    mask_id: int = 4
    eot_id: int = 3
    block_size: int = 2
    output_size: int = 4


class _FixedPQModel:
    config = _Config()

    def forward_sequence(self, ids, valid, *, positions=None):
        del valid, positions
        p = torch.tensor([0.10, 0.30, 0.60, 0.0], device=ids.device)
        q = torch.tensor([0.60, 0.30, 0.10, 0.0], device=ids.device)
        probabilities = torch.where(ids[..., None].eq(self.config.mask_id), q, p)
        return probabilities.log()


class _AlwaysEOTModel:
    config = _Config()

    def forward_sequence(self, ids, valid, *, positions=None):
        del valid, positions
        logits = torch.full((*ids.shape, 4), -torch.inf, device=ids.device)
        logits[..., self.config.eot_id] = 0
        return logits


def test_full_replay_pq_correction_recovers_clean_distribution() -> None:
    batch = 16_384
    prompt = torch.zeros((batch, 2), dtype=torch.long)
    valid = torch.ones_like(prompt, dtype=torch.bool)
    generator = torch.Generator().manual_seed(1234)
    result = generate_idlm_full_replay(
        _FixedPQModel(),
        prompt,
        valid,
        stride=2,
        max_new_tokens=2,
        generator=generator,
    )
    second = result.ids[:, 3]
    frequencies = torch.bincount(second, minlength=4).float() / batch
    torch.testing.assert_close(
        frequencies[:3], torch.tensor([0.10, 0.30, 0.60]), atol=0.02, rtol=0
    )
    assert result.iterations == 1
    assert result.model_forwards == 2


def test_full_replay_truncates_atomically_at_eot() -> None:
    prompt = torch.tensor([[0], [1]])
    valid = torch.ones_like(prompt, dtype=torch.bool)
    result = generate_idlm_full_replay(
        _AlwaysEOTModel(), prompt, valid, stride=2, max_new_tokens=8
    )
    assert result.lengths.tolist() == [2, 2]
    assert result.finished.tolist() == [True, True]
    assert result.ids[:, 1].tolist() == [3, 3]
    assert result.generated_valid.sum(1).tolist() == [1, 1]
    assert result.eligible_proposals == 0
    assert result.accepted_proposals == 0

    aligned = generate_idlm_fused_replay(
        _AlwaysEOTModel(),
        torch.tensor([[0, 1], [1, 0]]),
        torch.ones((2, 2), dtype=torch.bool),
        stride=2,
        max_new_tokens=8,
    )
    assert aligned.finished.tolist() == [True, True]
    assert aligned.physical_proposal_slots == 2
    assert aligned.eligible_proposals == 0


def test_max_new_tokens_is_relative_to_each_ragged_prompt() -> None:
    prompt = torch.tensor([[0, 1, 2], [0, 5, 5]])
    valid = torch.tensor([[True, True, True], [True, False, False]])
    result = generate_idlm_full_replay(
        _FixedPQModel(), prompt, valid, stride=2, max_new_tokens=2
    )
    assert result.lengths.tolist() == [5, 3]
    assert result.generated_valid.sum(1).tolist() == [2, 2]


def test_fused_replay_matches_oracle_distribution_and_accounts_physical_work() -> None:
    batch = 16_384
    prompt = torch.zeros((batch, 2), dtype=torch.long)
    valid = torch.ones_like(prompt, dtype=torch.bool)
    result = generate_idlm_fused_replay(
        _FixedPQModel(),
        prompt,
        valid,
        stride=2,
        max_new_tokens=2,
        generator=torch.Generator().manual_seed(1234),
    )
    second = result.ids[:, 3]
    frequencies = torch.bincount(second, minlength=4).float() / batch
    torch.testing.assert_close(
        frequencies[:3], torch.tensor([0.10, 0.30, 0.60]), atol=0.02, rtol=0
    )
    assert result.model_forwards == 2
    assert result.fused_2n_minus_1_forwards == 1
    assert result.physical_proposal_slots == batch * 3
    assert result.eligible_proposals == batch
    assert 0 <= result.accepted_proposals <= result.eligible_proposals
    assert not result.cache_backed


class _AlwaysZeroModel:
    config = _Config()

    def forward_sequence(self, ids, valid, *, positions=None):
        del valid, positions
        logits = torch.full((*ids.shape, 4), -torch.inf, device=ids.device)
        logits[..., 0] = 0
        return logits


def test_fused_replay_has_exact_point_mass_parity_with_two_pass_oracle() -> None:
    prompt = torch.tensor([[1, 0], [2, 0]])
    valid = torch.ones_like(prompt, dtype=torch.bool)
    oracle = generate_idlm_full_replay(
        _AlwaysZeroModel(), prompt, valid, stride=2, max_new_tokens=5
    )
    fused = generate_idlm_fused_replay(
        _AlwaysZeroModel(), prompt, valid, stride=2, max_new_tokens=5
    )
    torch.testing.assert_close(fused.ids, oracle.ids)
    torch.testing.assert_close(fused.valid, oracle.valid)
    assert fused.lengths.tolist() == oracle.lengths.tolist()
    assert fused.model_forwards < oracle.model_forwards


@dataclass(frozen=True)
class _Stride4Config(_Config):
    block_size: int = 4


class _RecordingStride4Model(_AlwaysZeroModel):
    config = _Stride4Config()

    def __init__(self) -> None:
        self.inputs = []

    def forward_sequence(self, ids, valid, *, positions=None):
        self.inputs.append((ids.clone(), valid.clone()))
        return super().forward_sequence(ids, valid, positions=positions)


def test_ragged_prompt_uses_clean_ar_alignment_before_trained_stride_masks() -> None:
    model = _RecordingStride4Model()
    prompt = torch.tensor([[1, 2, 0]])
    valid = torch.ones_like(prompt, dtype=torch.bool)
    result = generate_idlm_fused_replay(
        model, prompt, valid, stride=4, max_new_tokens=2
    )
    assert result.lengths.tolist() == [5]
    assert len(model.inputs) == 2
    first_ids, first_valid = model.inputs[0]
    assert not first_ids[first_valid].eq(model.config.mask_id).any()
    second_ids, second_valid = model.inputs[1]
    assert second_ids[second_valid].eq(model.config.mask_id).sum().item() == 3
    assert result.physical_proposal_slots == 3
    assert result.eligible_proposals == 0


def test_serving_stride_must_equal_checkpoint_trained_stride() -> None:
    prompt = torch.tensor([[0]])
    valid = torch.ones_like(prompt, dtype=torch.bool)
    with pytest.raises(ValueError, match="differs from trained stride"):
        generate_idlm_fused_replay(
            _AlwaysZeroModel(), prompt, valid, stride=4, max_new_tokens=2
        )
