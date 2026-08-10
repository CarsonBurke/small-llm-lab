"""CPU contracts for the versioned byte-diffusion configuration."""

from __future__ import annotations

import pytest

from pretraining.byte_diffusion.config import (
    AtomicVocabulary,
    ByteDiffusionConfig,
    CorruptionConfig,
)


def test_atomic_vocabulary_separates_outputs_mask_and_padding() -> None:
    vocab = AtomicVocabulary()
    assert vocab.output_size == 261
    assert vocab.input_size == 263
    assert vocab.eot_id == 256
    assert vocab.mask_id == vocab.output_size
    assert vocab.pad_id == vocab.output_size + 1


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"mask_id": 260}, "MASK"),
        ({"pad_id": 264}, "PAD"),
        ({"eot_id": 261}, "EOT"),
        ({"byte_values": 255}, "byte_values"),
    ],
)
def test_atomic_vocabulary_rejects_ambiguous_layouts(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        AtomicVocabulary(**overrides)


def test_default_and_tiny_configs_preserve_fixed_stride_contract() -> None:
    production = ByteDiffusionConfig()
    assert production.schema_version == 4
    assert production.decoder_prefix_window == 512
    assert production.decoder_branch_attention == "shared_flex"
    assert production.patch_stride == 4
    assert production.production_parameter_target == 23_011_074
    assert production.to_dict()["vocab"]["pad_id"] == 262

    tiny = ByteDiffusionConfig.tiny()
    assert tiny.local_dim == 32
    assert tiny.global_dim == 64
    assert tiny.patch_stride == production.patch_stride
    with pytest.raises(ValueError, match="closed parameter target"):
        _ = tiny.production_parameter_target


def test_configs_fail_closed_on_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="local_dim"):
        ByteDiffusionConfig.tiny(local_dim=30, local_heads=4)
    with pytest.raises(ValueError, match="fixed stride four"):
        ByteDiffusionConfig.tiny(patch_stride=2)
    with pytest.raises(ValueError, match="positive patch multiple"):
        CorruptionConfig(canvas_length=130)
    with pytest.raises(ValueError, match="branches_per_row"):
        CorruptionConfig(branches_per_row=0)
    with pytest.raises(ValueError, match="ngram_hash"):
        ByteDiffusionConfig.tiny(ngram_hash="bad")
    with pytest.raises(ValueError, match="ngram_table_sharing"):
        ByteDiffusionConfig.tiny(ngram_table_sharing="bad")
    with pytest.raises(ValueError, match="decoder_conditioning"):
        ByteDiffusionConfig.tiny(decoder_conditioning="bad")
    with pytest.raises(ValueError, match="global_ffn_kind"):
        ByteDiffusionConfig.tiny(global_ffn_kind="bad")
