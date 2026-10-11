"""Static intervention masks; no LM or CUDA execution."""
import pytest
import torch
from scripts.evaluate_state_flow import source_mask


@pytest.mark.parametrize("sources,width", [(2, 128), (6, 768), (6, 1536)])
def test_each_owner_keeps_exactly_its_fixed_source_channels(sources, width):
    masks = torch.stack([source_mask(width, sources, owner, "cpu") for owner in range(sources)])
    assert masks.dtype == torch.bool
    assert (masks.sum(0) == 1).all()
    for owner, mask in enumerate(masks):
        start = owner * (width // sources)
        expected = torch.zeros(width, dtype=torch.bool)
        expected[start:start + width // sources] = True
        assert torch.equal(mask, expected)


@pytest.mark.parametrize("width,sources,owner", [(0, 6, 0), (128, 0, 0), (128, 3, 0), (128, 2, 2), (128, 2, -1)])
def test_invalid_source_layout_is_rejected(width, sources, owner):
    with pytest.raises(ValueError):
        source_mask(width, sources, owner, "cpu")


def test_lattice_mask_keeps_lower_sources_and_removes_past_token_feedback():
    for owner in range(6):
        mask = source_mask(768, 6, owner, "cpu", include_lower=True).view(6, 128)
        assert mask[:owner + 1].all()
        assert not mask[owner + 1:].any()
