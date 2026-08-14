from __future__ import annotations

import torch

from pretraining.byte_diffusion.idlm import make_training_layout
from pretraining.byte_diffusion.idlm_model import (
    IDLMModel,
    IDLMModelConfig,
    idlm_dense_attention_mask,
)


def _layout(clean: torch.Tensor, valid: torch.Tensor, segments: torch.Tensor | None = None):
    return make_training_layout(
        clean,
        valid,
        mask_id=261,
        pad_id=262,
        block_size=4,
        segment_ids=segments,
    )


def test_default_idlm_cell_has_closed_parameter_count_and_mask_only_cue() -> None:
    model = IDLMModel()
    assert model.parameter_count() == 23_017_984
    names = tuple(name for name, _ in model.named_parameters())
    assert not any("mode" in name or "timestep" in name or "self_condition" in name for name in names)


def test_physical_order_does_not_change_idlm_logits() -> None:
    torch.manual_seed(11)
    model = IDLMModel(IDLMModelConfig.tiny()).eval()
    clean = torch.tensor([[7, 8, 9, 10, 11, 12, 13, 256]])
    valid = torch.ones_like(clean, dtype=torch.bool)
    layout = _layout(clean, valid)

    proposal_clean = model.forward_layout(
        layout,
        allow_dense_reference=True,
        physical_order="proposal_clean",
    )
    clean_proposal = model.forward_layout(
        layout,
        allow_dense_reference=True,
        physical_order="clean_proposal",
    )
    torch.testing.assert_close(
        proposal_clean.proposal_logits, clean_proposal.proposal_logits
    )
    torch.testing.assert_close(proposal_clean.clean_logits, clean_proposal.clean_logits)


def test_combined_clean_anchor_matches_ordinary_causal_forward() -> None:
    torch.manual_seed(12)
    model = IDLMModel(IDLMModelConfig.tiny()).eval()
    clean = torch.tensor([[2, 3, 5, 7, 11, 13, 17, 256]])
    valid = torch.ones_like(clean, dtype=torch.bool)
    layout = _layout(clean, valid)

    combined = model.forward_layout(layout, allow_dense_reference=True)
    causal = model.forward_sequence(clean, valid)
    torch.testing.assert_close(combined.clean_logits, causal)


def test_packed_isd_sequence_path_matches_dense_causal_oracle() -> None:
    model = IDLMModel(IDLMModelConfig.tiny()).eval()
    ids = torch.tensor([[1, 2, 3, 4], [7, 8, 262, 262]])
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    positions = torch.arange(4)[None].expand_as(ids)
    dense = model.forward_sequence(ids, valid, positions=positions)
    packed = model._forward_sequence_packed(
        ids, valid, positions, allow_dense_reference=True
    )
    torch.testing.assert_close(packed[valid], dense[valid])
    torch.testing.assert_close(packed[~valid], torch.zeros_like(packed[~valid]))


def test_current_clean_block_cannot_leak_into_its_proposal_logits() -> None:
    torch.manual_seed(13)
    model = IDLMModel(IDLMModelConfig.tiny()).eval()
    clean = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 256]])
    valid = torch.ones_like(clean, dtype=torch.bool)
    baseline = model.forward_layout(_layout(clean, valid), allow_dense_reference=True)

    changed = clean.clone()
    changed[:, 4:] = torch.tensor([90, 91, 92, 256])
    perturbed = model.forward_layout(_layout(changed, valid), allow_dense_reference=True)
    torch.testing.assert_close(
        baseline.proposal_logits[:, 4:8], perturbed.proposal_logits[:, 4:8]
    )


def test_aligned_serving_masks_match_the_trained_proposal_topology_exactly() -> None:
    torch.manual_seed(131)
    model = IDLMModel(IDLMModelConfig.tiny(block_size=4)).eval()
    clean = torch.tensor([[1, 2, 3, 4, 50, 51, 52, 53]])
    valid = torch.ones_like(clean, dtype=torch.bool)
    training = model.forward_layout(
        make_training_layout(
            clean,
            valid,
            mask_id=model.config.mask_id,
            pad_id=model.config.pad_id,
            block_size=4,
        ),
        allow_dense_reference=True,
    )
    serving_ids = torch.cat(
        (clean[:, :4], torch.full((1, 4), model.config.mask_id, dtype=torch.long)),
        dim=1,
    )
    serving = model.forward_sequence(serving_ids, torch.ones_like(serving_ids, dtype=torch.bool))
    torch.testing.assert_close(serving[:, 4:8], training.proposal_logits[:, 4:8])


def test_packed_documents_have_no_cross_document_leakage() -> None:
    torch.manual_seed(14)
    model = IDLMModel(IDLMModelConfig.tiny()).eval()
    clean = torch.tensor([[1, 2, 3, 256, 20, 21, 22, 256]])
    valid = torch.ones_like(clean, dtype=torch.bool)
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    baseline = model.forward_layout(
        _layout(clean, valid, segments), allow_dense_reference=True
    )

    changed = clean.clone()
    changed[:, 4:] = torch.tensor([100, 101, 102, 256])
    perturbed = model.forward_layout(
        _layout(changed, valid, segments), allow_dense_reference=True
    )
    torch.testing.assert_close(
        baseline.clean_logits[:, :4], perturbed.clean_logits[:, :4]
    )
    torch.testing.assert_close(
        baseline.proposal_logits[:, :4], perturbed.proposal_logits[:, :4]
    )


def test_window512_is_explicitly_different_from_full_prefix() -> None:
    clean = torch.arange(520)[None].remainder(255).to(torch.long)
    valid = torch.ones_like(clean, dtype=torch.bool)
    layout = _layout(clean, valid)
    exact = idlm_dense_attention_mask(layout, clean_prefix_window=None)
    windowed = idlm_dense_attention_mask(layout, clean_prefix_window=512)
    # Final clean query sees the first clean key under the paper mask only.
    assert exact[0, 2 * 520 - 1, 520]
    assert not windowed[0, 2 * 520 - 1, 520]
