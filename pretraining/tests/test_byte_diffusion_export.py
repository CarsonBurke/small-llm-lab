from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
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
    load_embedded_entropy_patcher,
    parse_artifact,
    quantize_int8,
    quantize_int4,
)
from pretraining.byte_diffusion.data import AtomicIdManifest
from pretraining.byte_diffusion.duo_model import DuoModel
from pretraining.byte_diffusion.patching import (
    CausalEntropyPatcher,
    EntropyPatchConfig,
    HashedNgramEntropyConfig,
    HashedNgramEntropyModel,
)
from pretraining.byte_diffusion.variable_patching import (
    ENTROPY_DATASET_SCHEMA,
    PATCHING_POLICY_SCHEMA,
)
from scripts.export_byte_diffusion import checkpoint_patcher_artifact
from scripts.export_byte_duo import cached_inference_smoke


def _entropy_patcher_artifact(*, max_patch_size: int = 4) -> bytes:
    patcher = CausalEntropyPatcher(
        HashedNgramEntropyModel(
            HashedNgramEntropyConfig(vocab_size=261, table_size=8)
        ),
        EntropyPatchConfig(max_patch_size=max_patch_size),
    )
    return patcher.to_bytes()


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


def test_entropy_artifact_is_self_contained_counted_and_hash_authenticated() -> None:
    config = ByteDiffusionConfig.tiny()
    model = ByteDiffusionModel(config)
    patcher_artifact = _entropy_patcher_artifact(max_patch_size=5)
    fixed = build_artifact(model, config, code_bytes=1234)
    assert fixed == build_artifact(
        model, config, code_bytes=1234, entropy_patcher=None
    )
    artifact = build_artifact(
        model,
        config,
        code_bytes=1234,
        entropy_patcher=patcher_artifact,
    )
    report = artifact_size_report(
        model,
        config,
        code_bytes=1234,
        entropy_patcher=patcher_artifact,
    )
    metadata, _ = parse_artifact(artifact)
    info = metadata["entropy_patcher"]

    assert "entropy_patcher" not in parse_artifact(fixed)[0]
    assert len(artifact) == report.artifact_bytes
    assert report.complete_bytes == len(artifact) + 1234
    assert len(artifact) > len(fixed) + len(patcher_artifact)
    assert info["payload_bytes"] == len(patcher_artifact)
    assert info["sha256"] == hashlib.sha256(patcher_artifact).hexdigest()
    assert info["max_patch_size"] == 5
    restored = load_embedded_entropy_patcher(artifact)
    assert restored is not None
    assert restored.sha256 == hashlib.sha256(patcher_artifact).hexdigest()


def test_entropy_artifact_rejects_patcher_tampering_and_counts_it_against_cap() -> None:
    config = ByteDiffusionConfig.tiny()
    model = ByteDiffusionModel(config)
    patcher_artifact = _entropy_patcher_artifact()
    artifact = build_artifact(model, config, entropy_patcher=patcher_artifact)
    metadata, _ = parse_artifact(artifact)
    info = metadata["entropy_patcher"]
    metadata_bytes = struct.unpack_from("<Q", artifact, len(MAGIC))[0]
    payload_start = len(MAGIC) + 8 + metadata_bytes
    corrupted = bytearray(artifact)
    corrupted[payload_start + int(info["payload_offset"])] ^= 1
    with pytest.raises(ValueError, match="sha256 mismatch"):
        parse_artifact(bytes(corrupted))

    report = artifact_size_report(
        model, config, entropy_patcher=patcher_artifact
    )
    with pytest.raises(ValueError, match="exceeding"):
        build_artifact(
            model,
            config,
            code_bytes=ARTIFACT_CAP_BYTES - report.artifact_bytes + 1,
            entropy_patcher=patcher_artifact,
        )


def test_final_entropy_artifact_requires_matching_embedded_patcher() -> None:
    config = ByteDiffusionConfig.tiny()
    model = ByteDiffusionModel(config)
    patcher_artifact = _entropy_patcher_artifact()
    patcher_sha256 = hashlib.sha256(patcher_artifact).hexdigest()
    metrics = {
        "bpb": 1.23,
        "literal_bytes": 100,
        "encoding_plan": {
            "policy": "uniform_int4",
            "group_size": 64,
            "fallbacks": {},
        },
    }
    provenance = {
        "patching_policy": "causal_entropy_v1",
        "entropy_patcher_sha256": patcher_sha256,
    }

    with pytest.raises(ValueError, match="require an embedded"):
        build_artifact(
            model,
            config,
            post_quantization_metrics=metrics,
            provenance=provenance,
            require_evaluation=True,
        )
    with pytest.raises(ValueError, match="provenance disagrees"):
        build_artifact(
            model,
            config,
            post_quantization_metrics=metrics,
            provenance={**provenance, "entropy_patcher_sha256": "0" * 64},
            entropy_patcher=patcher_artifact,
            require_evaluation=True,
        )
    artifact = build_artifact(
        model,
        config,
        post_quantization_metrics=metrics,
        provenance=provenance,
        entropy_patcher=patcher_artifact,
        require_evaluation=True,
    )
    assert load_embedded_entropy_patcher(artifact) is not None


def test_export_loads_patcher_only_from_checkpoint_pinned_dataset(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    patcher_artifact = _entropy_patcher_artifact()
    patcher = CausalEntropyPatcher.from_bytes(patcher_artifact)
    artifact_path = dataset / "entropy-patcher.bdpatch"
    artifact_path.write_bytes(patcher_artifact)
    manifest = {
        "schema": ENTROPY_DATASET_SCHEMA,
        "packing": {"patch_stride": None},
        "patching": {
            "schema": PATCHING_POLICY_SCHEMA,
            "name": "causal_entropy_v1",
            "patcher_artifact": {
                "path": artifact_path.name,
                "sha256": hashlib.sha256(patcher_artifact).hexdigest(),
                "bytes": len(patcher_artifact),
            },
            "entropy_model_config": asdict(patcher.model.config),
            "boundary_config": asdict(patcher.config),
            "max_patch_size": patcher.config.max_patch_size,
        },
    }
    payload_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["payload_sha256"] = payload_sha256
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    checkpoint = {
        "run_contract": {
            "patching_policy": "causal_entropy_v1",
            "corruption": {"canvas_length": 4},
        },
        "dataset_provenance": {"payload_sha256": payload_sha256},
    }

    assert checkpoint_patcher_artifact(checkpoint, dataset) == patcher_artifact
    checkpoint["dataset_provenance"]["payload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint-pinned"):
        checkpoint_patcher_artifact(checkpoint, dataset)


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
    assert "mode_embedding.weight" not in tensors
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
        assert mixed.parameter_count == 29_403_138
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


def test_duo_export_cached_inference_smoke_matches_full_forward() -> None:
    result = cached_inference_smoke(DuoModel(ByteDiffusionConfig.tiny()).eval())

    assert result["finite"] is True
    assert result["full_vs_cached_max_abs_error"] < 1e-5


def test_final_duo_artifact_requires_its_conditional_canvas_metric_not_fake_bpb() -> None:
    config = ByteDiffusionConfig.tiny()
    model = DuoModel(config)
    provenance = {
        "architecture": "byte_duo_uniform_state_diffusion",
        "checkpoint_step": 2000,
        "dataset_payload_sha256": "b" * 64,
    }
    metrics = {
        "conditional_canvas_nelbo_nats_per_atom": 1.75,
        "targets": 4096,
        "quantization_delta_bits_per_atom": 0.04,
        "max_quantization_delta_bits_per_atom": 0.05,
        "cached_inference_smoke": {
            "finite": True,
            "full_vs_cached_max_abs_error": 0.0,
        },
        "encoding_plan": {
            "policy": "uniform_int4",
            "group_size": 64,
            "fallbacks": {},
        },
    }
    artifact = build_artifact(
        model,
        config,
        post_quantization_metrics=metrics,
        provenance=provenance,
        require_evaluation=True,
    )
    metadata, _ = parse_artifact(artifact)
    assert metadata["post_quantization_metrics"] == metrics
    with pytest.raises(ValueError, match="conditional-canvas"):
        build_artifact(
            model,
            config,
            post_quantization_metrics={
                **metrics,
                "conditional_canvas_nelbo_nats_per_atom": float("nan"),
            },
            provenance=provenance,
            require_evaluation=True,
        )
    with pytest.raises(ValueError, match="degradation threshold"):
        build_artifact(
            model,
            config,
            post_quantization_metrics={
                **metrics,
                "quantization_delta_bits_per_atom": 0.051,
            },
            provenance=provenance,
            require_evaluation=True,
        )
    with pytest.raises(ValueError, match="cached inference parity"):
        build_artifact(
            model,
            config,
            post_quantization_metrics={
                **metrics,
                "cached_inference_smoke": {
                    "finite": True,
                    "full_vs_cached_max_abs_error": 1e-3,
                },
            },
            provenance=provenance,
            require_evaluation=True,
        )


def test_parameter_cap_fails_closed() -> None:
    oversized = nn.Embedding(PARAMETER_CAP + 1, 1)
    try:
        enforce_parameter_cap(oversized)
    except ValueError as error:
        assert "exceeding" in str(error)
    else:
        raise AssertionError("oversized model was accepted")
