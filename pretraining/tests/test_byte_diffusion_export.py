from __future__ import annotations

import json
import struct

import pytest
import torch
from torch import nn

from pretraining.byte_diffusion import ByteDiffusionConfig, ByteDiffusionModel
from pretraining.byte_diffusion.export import (
    ARTIFACT_CAP_BYTES,
    MAGIC,
    PARAMETER_CAP,
    artifact_size_report,
    build_artifact,
    decode_float16,
    dequantize_int8,
    dequantize_int4,
    encode_float16,
    enforce_parameter_cap,
    load_artifact,
    parse_artifact,
    quantize_int8,
    quantize_int4,
)
from pretraining.byte_diffusion.data import AtomicIdManifest


def test_nibble_roundtrip_odd_tensor_has_declared_error_bound() -> None:
    values = torch.tensor([-1.0, -0.5, 0.0, 0.4, 1.0])
    quantized = quantize_int4(values, group_size=4)
    restored = dequantize_int4(quantized)
    scales = torch.frombuffer(bytearray(quantized.scales), dtype=torch.float16).float()
    assert torch.all((restored - values).abs() <= scales.repeat_interleave(4)[:5] / 2 + 1e-3)


def test_int8_roundtrip_has_declared_error_bound() -> None:
    values = torch.linspace(-1.0, 1.0, 67)
    quantized = quantize_int8(values, group_size=16)
    restored = dequantize_int8(quantized)
    scales = torch.frombuffer(bytearray(quantized.scales), dtype=torch.float16).float()
    error_bound = scales.repeat_interleave(16)[: values.numel()] / 2 + 1e-3
    assert torch.all((restored - values).abs() <= error_bound)


def test_float16_encoding_is_bit_exact_at_its_declared_precision() -> None:
    values = torch.tensor([-1000.25, -0.0, 0.1, 1.5, 65504.0], dtype=torch.float32)
    encoded = encode_float16(values)
    restored = decode_float16(encoded)
    assert torch.equal(restored, values.to(torch.float16).to(torch.float32))


def test_parser_rejects_non_object_and_negative_tensor_metadata() -> None:
    list_metadata = b"[]"
    with pytest.raises(ValueError, match="JSON object"):
        parse_artifact(MAGIC + struct.pack("<Q", len(list_metadata)) + list_metadata)

    negative_metadata = json.dumps(
        {
            "schema": 3,
            "tensors": {
                "bad": {
                    "shape": [-1],
                    "count": -1,
                    "encoding": "group_int4",
                    "group_size": 64,
                    "packed_offset": 0,
                    "packed_bytes": 0,
                    "scale_offset": 0,
                    "scale_bytes": 0,
                }
            },
        },
        separators=(",", ":"),
    ).encode()
    artifact = MAGIC + struct.pack("<Q", len(negative_metadata)) + negative_metadata
    with pytest.raises(ValueError, match="negative"):
        parse_artifact(artifact)


def test_small_artifact_strict_roundtrip() -> None:
    config = ByteDiffusionConfig.tiny()
    source = ByteDiffusionModel(config)
    artifact = build_artifact(source, config, group_size=64, code_bytes=1234)
    metadata, tensors = parse_artifact(artifact)
    assert metadata["code_bytes"] == 1234
    assert metadata["encoding_policy"] == "uniform_int4"
    assert {tensor.encoding for tensor in tensors.values()} == {"group_int4"}
    assert set(tensors) == set(dict(source.named_parameters()))
    target = ByteDiffusionModel(config)
    load_artifact(target, artifact)
    for name, parameter in target.named_parameters():
        assert torch.isfinite(parameter).all(), name


def test_mixed_artifact_records_and_exactly_decodes_each_encoding() -> None:
    config = ByteDiffusionConfig.tiny(
        ngram_enabled=True,
        ngram_table_size=64,
        ngram_rank=4,
        ngram_orders=(3,),
    )
    source = ByteDiffusionModel(config)
    artifact = build_artifact(source, config, encoding_policy="mixed_sensitive")
    report = artifact_size_report(
        source, config, encoding_policy="mixed_sensitive"
    )
    metadata, tensors = parse_artifact(artifact)

    assert len(artifact) == report.artifact_bytes
    assert metadata["encoding_policy"] == "mixed_sensitive"
    assert tensors["embedding.weight"].encoding == "float16"
    assert tensors["mode_embedding.weight"].encoding == "float16"
    assert tensors["output.weight"].encoding == "float16"
    assert tensors["encoder.0.attention_norm.weight"].encoding == "float16"
    assert tensors["decoder.0.condition_gate"].encoding == "float16"
    assert tensors["ngrams.table.weight"].encoding == "group_int8"
    assert tensors["encoder.0.attention.qkv.weight"].encoding == "group_int4"
    assert metadata["encoding_fallbacks"] == {}

    target = ByteDiffusionModel(config)
    load_artifact(target, artifact)
    source_parameters = dict(source.named_parameters())
    target_parameters = dict(target.named_parameters())
    for name, encoded in tensors.items():
        if encoded.encoding != "float16":
            continue
        expected = source_parameters[name].detach().to(torch.float16).to(torch.float32)
        assert torch.equal(target_parameters[name].detach(), expected), name


def test_mixed_artifact_uses_int8_for_per_order_ngram_tables() -> None:
    config = ByteDiffusionConfig.tiny(
        ngram_enabled=True,
        ngram_table_sharing="per_order",
        ngram_table_size=64,
        ngram_rank=4,
        ngram_orders=(3, 4),
    )
    model = ByteDiffusionModel(config)
    artifact = build_artifact(model, config, encoding_policy="mixed_sensitive")
    _, tensors = parse_artifact(artifact)
    ngram_encodings = {
        name: tensor.encoding
        for name, tensor in tensors.items()
        if name.startswith("ngrams.tables.")
    }
    assert ngram_encodings == {
        "ngrams.tables.0.weight": "group_int8",
        "ngrams.tables.1.weight": "group_int8",
    }


@pytest.mark.parametrize("global_layers", [9, 12])
def test_production_artifact_size_accounting_compares_uniform_and_mixed(
    global_layers: int,
) -> None:
    config = ByteDiffusionConfig(global_layers=global_layers)
    model = ByteDiffusionModel(config)
    # A compressed submission wrapper must fit in this explicit allowance for
    # the near-capacity 12-layer mixed arm.
    code_bytes = 100_000
    uniform = artifact_size_report(
        model,
        config,
        code_bytes=code_bytes,
        encoding_policy="uniform_int4",
    )
    mixed = artifact_size_report(
        model,
        config,
        code_bytes=code_bytes,
        encoding_policy="mixed_sensitive",
    )

    assert uniform.complete_bytes < ARTIFACT_CAP_BYTES
    assert mixed.complete_bytes < ARTIFACT_CAP_BYTES
    assert mixed.complete_bytes > uniform.complete_bytes
    assert uniform.encoding_parameter_counts == {
        "group_int4": uniform.parameter_count,
        "group_int8": 0,
        "float16": 0,
    }
    assert mixed.encoding_parameter_counts["float16"] > 0
    if global_layers == 9:
        assert mixed.encoding_parameter_counts["group_int8"] == 524_288
        assert mixed.encoding_fallbacks == {}
    else:
        assert mixed.parameter_count == 29_403_906
        assert mixed.encoding_parameter_counts["group_int8"] == 0
        assert mixed.encoding_fallbacks == {
            "ngrams.table.weight": "group_int8_to_group_int4_artifact_budget"
        }
        assert mixed.headroom_bytes > 0
        assert mixed.headroom_bytes < 50_000


def test_twelve_layer_mixed_artifact_declares_its_code_size_limit() -> None:
    config = ByteDiffusionConfig(global_layers=12)
    model = ByteDiffusionModel(config)
    report = artifact_size_report(
        model, config, encoding_policy="mixed_sensitive"
    )
    assert report.artifact_bytes < ARTIFACT_CAP_BYTES
    assert report.headroom_bytes == ARTIFACT_CAP_BYTES - report.artifact_bytes
    assert 100_000 < report.headroom_bytes < 150_000

    uncompressed_code = 500_000
    oversized = artifact_size_report(
        model,
        config,
        code_bytes=uncompressed_code,
        encoding_policy="mixed_sensitive",
    )
    assert oversized.complete_bytes > ARTIFACT_CAP_BYTES
    with pytest.raises(ValueError, match="exceeding"):
        # The size plan fails before any 29M-parameter quantization work begins.
        build_artifact(
            model,
            config,
            code_bytes=uncompressed_code,
            encoding_policy="mixed_sensitive",
        )


def test_final_artifact_binds_atomic_manifest_and_dequantized_metrics() -> None:
    config = ByteDiffusionConfig.tiny()
    model = ByteDiffusionModel(config)
    manifest = AtomicIdManifest.reference()
    with pytest.raises(ValueError, match="post-quantization"):
        build_artifact(model, config, require_evaluation=True)
    with pytest.raises(ValueError, match="provenance"):
        build_artifact(
            model,
            config,
            post_quantization_metrics={"bpb": 1.23, "literal_bytes": 100},
            require_evaluation=True,
        )
    provenance = {
        "checkpoint_step": 2000,
        "validation_dataset_sha256": "a" * 64,
    }
    with pytest.raises(ValueError, match="encoding_plan"):
        build_artifact(
            model,
            config,
            post_quantization_metrics={"bpb": 1.23, "literal_bytes": 100},
            provenance=provenance,
            require_evaluation=True,
        )
    metrics = {
        "bpb": 1.23,
        "literal_bytes": 100,
        "encoding_plan": {
            "policy": "uniform_int4",
            "group_size": 64,
            "fallbacks": {},
        },
    }
    artifact = build_artifact(
        model,
        config,
        atomic_manifest=manifest,
        post_quantization_metrics=metrics,
        provenance=provenance,
        require_evaluation=True,
    )
    metadata, _ = parse_artifact(artifact)
    assert metadata["atomic_manifest_sha256"] == manifest.sha256
    assert metadata["atomic_manifest"] == manifest.to_dict()
    assert metadata["post_quantization_metrics"] == metrics
    assert metadata["provenance"] == provenance


def test_parameter_cap_fails_closed() -> None:
    oversized = nn.Embedding(PARAMETER_CAP + 1, 1)
    try:
        enforce_parameter_cap(oversized)
    except ValueError as error:
        assert "exceeding" in str(error)
    else:
        raise AssertionError("oversized model was accepted")
