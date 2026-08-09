from __future__ import annotations

import torch
import pytest
from torch import nn

from pretraining.byte_diffusion import ByteDiffusionConfig, ByteDiffusionModel
from pretraining.byte_diffusion.export import (
    build_artifact,
    dequantize_int4,
    enforce_parameter_cap,
    load_artifact,
    parse_artifact,
    quantize_int4,
)
from pretraining.byte_diffusion.data import AtomicIdManifest


def test_nibble_roundtrip_odd_tensor_has_declared_error_bound() -> None:
    values = torch.tensor([-1.0, -0.5, 0.0, 0.4, 1.0])
    quantized = quantize_int4(values, group_size=4)
    restored = dequantize_int4(quantized)
    scales = torch.frombuffer(bytearray(quantized.scales), dtype=torch.float16).float()
    assert torch.all((restored - values).abs() <= scales.repeat_interleave(4)[:5] / 2 + 1e-3)


def test_small_artifact_strict_roundtrip() -> None:
    config = ByteDiffusionConfig.tiny()
    source = ByteDiffusionModel(config)
    artifact = build_artifact(source, config, group_size=64, code_bytes=1234)
    metadata, tensors = parse_artifact(artifact)
    assert metadata["code_bytes"] == 1234
    assert set(tensors) == set(dict(source.named_parameters()))
    target = ByteDiffusionModel(config)
    load_artifact(target, artifact)
    for name, parameter in target.named_parameters():
        assert torch.isfinite(parameter).all(), name


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
    artifact = build_artifact(
        model,
        config,
        atomic_manifest=manifest,
        post_quantization_metrics={"bpb": 1.23, "literal_bytes": 100},
        provenance=provenance,
        require_evaluation=True,
    )
    metadata, _ = parse_artifact(artifact)
    assert metadata["atomic_manifest_sha256"] == manifest.sha256
    assert metadata["atomic_manifest"] == manifest.to_dict()
    assert metadata["post_quantization_metrics"] == {
        "bpb": 1.23,
        "literal_bytes": 100,
    }
    assert metadata["provenance"] == provenance


def test_parameter_cap_fails_closed() -> None:
    oversized = nn.Embedding(24_800_001, 1)
    try:
        enforce_parameter_cap(oversized)
    except ValueError as error:
        assert "exceeding" in str(error)
    else:
        raise AssertionError("oversized model was accepted")
