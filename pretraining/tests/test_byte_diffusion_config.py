"""CPU contracts for the versioned byte-diffusion configuration."""

from __future__ import annotations

import pytest

from pretraining.byte_diffusion.config import (
    FAST_BLT_ENTROPY_B4_COMPLETE_PAPER_RATIO_PARAMETER_TARGET,
    FAST_BLT_ENTROPY_B4_COMPLETE_PAPER_RATIO_PRESET,
    FAST_BLT_ENTROPY_B4_COMPLETE_PARAMETER_TARGET,
    FAST_BLT_ENTROPY_B4_COMPLETE_PRESET,
    AtomicVocabulary,
    ByteDiffusionConfig,
    CorruptionConfig,
    model_config_from_env,
)
from pretraining.byte_diffusion.model import ByteDiffusionModel


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
    assert production.schema_version == 5
    assert production.decoder_prefix_window == 512
    assert production.decoder_branch_attention == "shared_flex"
    assert production.patch_stride == 4
    assert production.duo_time_features == 64
    assert production.duo_time_condition_dim == 32
    assert production.duo_diffusion_atoms == 257
    assert production.duo_variable_length_probability == 0.01
    assert production.production_parameter_target == 23_010_306
    assert production.to_dict()["vocab"]["pad_id"] == 262

    tiny = ByteDiffusionConfig.tiny()
    assert tiny.local_dim == 32
    assert tiny.global_dim == 64
    assert tiny.patch_stride == production.patch_stride
    with pytest.raises(ValueError, match="closed parameter target"):
        _ = tiny.production_parameter_target


def test_fast_blt_entropy_b4_complete_is_an_exact_named_model_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = ByteDiffusionConfig.fast_blt_entropy_b4_complete()
    monkeypatch.setenv(
        "BYTE_DIFFUSION_PRESET", FAST_BLT_ENTROPY_B4_COMPLETE_PRESET
    )

    observed = model_config_from_env()

    assert observed == expected
    assert (observed.encoder_layers, observed.global_layers, observed.decoder_layers) == (
        1,
        6,
        8,
    )
    assert (
        observed.encoder_ffn_dim,
        observed.global_ffn_dim,
        observed.decoder_ffn_dim,
    ) == (512, 512, 680)
    assert observed.ngram_table_size == 8_192
    assert observed.ngram_rank == 16
    assert observed.ngram_orders == (3, 4, 5, 6, 7, 8)
    assert observed.ngram_hash == "blt_prime"
    assert observed.ngram_factor_init == "scale_matched"
    assert observed.ngram_aggregation == "mean"
    assert len(observed.ngram_orders) + 1 == 7
    assert observed.ngram_table_sharing == "per_order"
    assert observed.decoder_conditioning == "split_cross_attention"
    assert observed.decoder_prefix_window is None
    assert not observed.output_tied
    assert observed.production_parameter_target == (
        FAST_BLT_ENTROPY_B4_COMPLETE_PARAMETER_TARGET
    )
    model = ByteDiffusionModel(observed)
    assert model.output.bias is None
    assert model.parameter_count() == FAST_BLT_ENTROPY_B4_COMPLETE_PARAMETER_TARGET


@pytest.mark.parametrize(
    ("environment", "value"),
    [
        ("BYTE_DIFFUSION_GLOBAL_LAYERS", "7"),
        ("BYTE_DIFFUSION_DECODER_LAYERS", "7"),
        ("BYTE_DIFFUSION_DECODER_FFN_DIM", "681"),
        ("BYTE_DIFFUSION_NGRAM_TABLE_SIZE", "32768"),
        ("BYTE_DIFFUSION_NGRAM_FACTOR_INIT", "weak"),
        ("BYTE_DIFFUSION_NGRAM_AGGREGATION", "sum"),
        ("BYTE_DIFFUSION_NGRAM_TABLE_SHARING", "shared"),
        ("BYTE_DIFFUSION_DECODER_CONDITIONING", "gated_projection"),
        ("BYTE_DIFFUSION_DECODER_PREFIX_WINDOW", "512"),
    ],
)
def test_fast_blt_entropy_b4_complete_rejects_model_drift(
    monkeypatch: pytest.MonkeyPatch, environment: str, value: str
) -> None:
    monkeypatch.setenv(
        "BYTE_DIFFUSION_PRESET", FAST_BLT_ENTROPY_B4_COMPLETE_PRESET
    )
    monkeypatch.setenv(environment, value)

    with pytest.raises(ValueError, match="model contract mismatch"):
        model_config_from_env()


def test_fast_blt_entropy_b4_complete_rejects_tiny_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "BYTE_DIFFUSION_PRESET", FAST_BLT_ENTROPY_B4_COMPLETE_PRESET
    )
    with pytest.raises(ValueError, match="full production architecture"):
        model_config_from_env(tiny=True)


def test_fast_blt_entropy_b4_complete_paper_ratio_is_an_isolated_named_ablation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete = ByteDiffusionConfig.fast_blt_entropy_b4_complete()
    expected = ByteDiffusionConfig.fast_blt_entropy_b4_complete_paper_ratio()
    monkeypatch.setenv(
        "BYTE_DIFFUSION_PRESET",
        FAST_BLT_ENTROPY_B4_COMPLETE_PAPER_RATIO_PRESET,
    )

    observed = model_config_from_env()

    assert observed == expected
    assert (observed.encoder_layers, observed.global_layers, observed.decoder_layers) == (
        1,
        10,
        2,
    )
    assert (
        observed.encoder_ffn_dim,
        observed.global_ffn_dim,
        observed.decoder_ffn_dim,
    ) == (512, 544, 680)
    changed_fields = {"global_layers", "decoder_layers", "global_ffn_dim"}
    assert {
        field: value
        for field, value in observed.to_dict().items()
        if field not in changed_fields
    } == {
        field: value
        for field, value in complete.to_dict().items()
        if field not in changed_fields
    }
    assert observed.production_parameter_target == (
        FAST_BLT_ENTROPY_B4_COMPLETE_PAPER_RATIO_PARAMETER_TARGET
    )
    model = ByteDiffusionModel(observed)
    assert model.output.bias is None
    assert model.parameter_count() == (
        FAST_BLT_ENTROPY_B4_COMPLETE_PAPER_RATIO_PARAMETER_TARGET
    )


@pytest.mark.parametrize(
    ("environment", "value"),
    [
        ("BYTE_DIFFUSION_ENCODER_LAYERS", "2"),
        ("BYTE_DIFFUSION_GLOBAL_LAYERS", "9"),
        ("BYTE_DIFFUSION_DECODER_LAYERS", "3"),
        ("BYTE_DIFFUSION_GLOBAL_FFN_DIM", "545"),
        ("BYTE_DIFFUSION_DECODER_FFN_DIM", "681"),
        ("BYTE_DIFFUSION_NGRAM_TABLE_SIZE", "32768"),
        ("BYTE_DIFFUSION_DECODER_CONDITIONING", "gated_projection"),
        ("BYTE_DIFFUSION_DECODER_PREFIX_WINDOW", "512"),
    ],
)
def test_fast_blt_entropy_b4_complete_paper_ratio_rejects_model_drift(
    monkeypatch: pytest.MonkeyPatch, environment: str, value: str
) -> None:
    monkeypatch.setenv(
        "BYTE_DIFFUSION_PRESET",
        FAST_BLT_ENTROPY_B4_COMPLETE_PAPER_RATIO_PRESET,
    )
    monkeypatch.setenv(environment, value)

    with pytest.raises(ValueError, match="model contract mismatch"):
        model_config_from_env()


def test_fast_blt_entropy_b4_complete_paper_ratio_rejects_tiny_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "BYTE_DIFFUSION_PRESET",
        FAST_BLT_ENTROPY_B4_COMPLETE_PAPER_RATIO_PRESET,
    )
    with pytest.raises(ValueError, match="full production architecture"):
        model_config_from_env(tiny=True)


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
    with pytest.raises(ValueError, match="architecture values"):
        ByteDiffusionConfig.tiny(ngram_table_size=0)
    with pytest.raises(ValueError, match="decoder_split_residual_scale"):
        ByteDiffusionConfig.tiny(decoder_split_residual_scale=0)
    with pytest.raises(ValueError, match="duo_time_features"):
        ByteDiffusionConfig.tiny(duo_time_features=63)
    for unsupported_atoms in (256, 258, 259, 260, 262):
        with pytest.raises(ValueError, match="duo_diffusion_atoms"):
            ByteDiffusionConfig.tiny(duo_diffusion_atoms=unsupported_atoms)
    assert (
        ByteDiffusionConfig.tiny(duo_diffusion_atoms=257).duo_diffusion_atoms == 257
    )
    assert (
        ByteDiffusionConfig.tiny(duo_diffusion_atoms=261).duo_diffusion_atoms == 261
    )
    with pytest.raises(ValueError, match="duo_variable_length_probability"):
        ByteDiffusionConfig.tiny(duo_variable_length_probability=1.01)


def test_duo_prior_support_and_variable_length_are_explicit_env_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BYTE_DUO_DIFFUSION_ATOMS", "261")
    monkeypatch.setenv("BYTE_DUO_VARIABLE_LENGTH_PROBABILITY", "0")
    config = model_config_from_env(tiny=True)
    assert config.duo_diffusion_atoms == 261
    assert config.duo_variable_length_probability == 0.0


def test_duo_prior_support_does_not_expand_with_custom_vocabulary() -> None:
    expanded = AtomicVocabulary(clean_specials=6, mask_id=262, pad_id=263)
    with pytest.raises(ValueError, match="supported posterior size"):
        ByteDiffusionConfig.tiny(vocab=expanded, duo_diffusion_atoms=262)

    shifted_eot = AtomicVocabulary(eot_id=257)
    with pytest.raises(ValueError, match="supported posterior size"):
        ByteDiffusionConfig.tiny(vocab=shifted_eot, duo_diffusion_atoms=258)
