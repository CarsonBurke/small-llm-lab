#!/usr/bin/env python3
"""Export and post-quantization-evaluate a completed Byte-Duo checkpoint.

GPU execution is a model workload and must be submitted through ``mlq``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.byte_diffusion.config import ByteDiffusionConfig
from pretraining.byte_diffusion.duo_model import DuoModel
from pretraining.byte_diffusion.export import build_artifact, load_artifact, parse_artifact, write_artifact
from pretraining.byte_diffusion.inference_duo import duo_entropy_clean_metadata
from pretraining.byte_diffusion.patching import CausalEntropyPatcher
from pretraining.byte_diffusion.training import load_data_directory, model_config_from_dict
from pretraining.byte_diffusion.training_duo import (
    PreparedDuoValidationBatch,
    prepare_duo_validation_inputs,
    validate_duo,
)
from scripts.train_byte_duo import (
    _validation_batches,
    complete_code_provenance,
    counted_duo_code_paths,
    source_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument(
        "--encoding-policy",
        choices=("uniform_int4", "mixed_sensitive"),
        default="uniform_int4",
    )
    parser.add_argument("--validation-rows", type=int, default=256)
    parser.add_argument("--validation-batch-size", type=int, default=24)
    parser.add_argument("--max-quantization-delta-bits", type=float, default=0.05)
    parser.add_argument("--cpu-reference", action="store_true")
    return parser.parse_args()


def checkpoint_entropy_patcher_artifact(
    payload: dict[str, object], data_path: Path
) -> bytes | None:
    """Recover only the dataset- and checkpoint-authenticated Duo patcher."""

    config = model_config_from_dict(payload["model_config"])
    training = payload.get("training")
    binding = training.get("dataset_patching") if isinstance(training, dict) else None
    if not isinstance(binding, dict):
        raise ValueError("Duo checkpoint omitted its dataset patching contract")
    if binding.get("name") != config.duo_clean_patching:
        raise ValueError("Duo checkpoint patching policy differs from model config")
    if config.duo_clean_patching == "fixed_stride_v1":
        if binding.get("patcher_sha256") is not None:
            raise ValueError("fixed-stride Duo checkpoint claims an entropy patcher")
        return None
    manifest = json.loads((data_path / "manifest.json").read_text())
    patching = manifest.get("patching")
    if not isinstance(patching, dict) or patching.get("name") != (
        "causal_entropy_v1"
    ):
        raise ValueError("Duo export dataset omitted causal entropy patching")
    artifact_info = patching.get("patcher_artifact")
    if not isinstance(artifact_info, dict):
        raise ValueError("Duo export dataset omitted its patcher artifact")
    relative = Path(str(artifact_info.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Duo entropy patcher path must stay within its dataset")
    dataset_root = data_path.resolve()
    artifact_path = (data_path / relative).resolve()
    if not artifact_path.is_relative_to(dataset_root):
        raise ValueError("Duo entropy patcher path escaped through a symlink")
    artifact = artifact_path.read_bytes()
    patcher = CausalEntropyPatcher.from_bytes(artifact)
    expected_sha256 = binding.get("patcher_sha256")
    if expected_sha256 != patcher.sha256 or artifact_info.get("sha256") != (
        patcher.sha256
    ):
        raise ValueError("Duo entropy patcher differs from authenticated provenance")
    if int(binding.get("max_patch_size", -1)) != patcher.config.max_patch_size:
        raise ValueError("Duo entropy patcher maximum size differs from checkpoint")
    return artifact


@torch.no_grad()
def cached_inference_smoke(
    model: DuoModel, entropy_patcher: CausalEntropyPatcher | None = None
) -> dict[str, object]:
    """Verify the artifact's deployed cached path against its full forward."""

    if (model.config.duo_clean_patching == "causal_entropy_v1") != (
        entropy_patcher is not None
    ):
        raise ValueError("cached smoke requires its policy-matched entropy patcher")
    stride = model.config.patch_stride
    full_resolution = (
        model.config.duo_mutable_topology == "full_resolution_decoder"
    )
    smoke_phase = 1 if full_resolution else 0
    clean_atoms = 2 * stride + smoke_phase
    clean_length = 3 * stride
    clean_ids = torch.arange(clean_length, dtype=torch.long).remainder(256)[None]
    clean_ids[:, clean_atoms:] = model.config.vocab.pad_id
    clean_valid = torch.arange(clean_length)[None] < clean_atoms
    document_ids = torch.zeros_like(clean_ids)
    positions = torch.arange(clean_length, dtype=torch.long)[None]
    noisy_ids = torch.arange(stride, dtype=torch.long)[None, None]
    branch_valid = torch.ones_like(noisy_ids, dtype=torch.bool)
    branch_start = clean_atoms
    if branch_start != int(clean_valid.sum()):
        raise AssertionError("cached smoke canvas must begin at the exact clean length")
    branch_starts = torch.full((1, 1), branch_start, dtype=torch.long)
    times = torch.full((1, 1), 0.5)
    clean_patch_metadata = (
        duo_entropy_clean_metadata(
            entropy_patcher, clean_ids, clean_valid, document_ids
        )
        if entropy_patcher is not None
        else None
    )
    full = model(
        clean_ids,
        clean_valid,
        document_ids,
        positions,
        noisy_ids,
        branch_valid,
        branch_starts,
        times,
        clean_patch_metadata=clean_patch_metadata,
        allow_synthetic_branch_suffix=True,
    ).branch_logits
    bank = model.prepare_clean_bank(
        clean_ids,
        clean_valid,
        document_ids,
        positions,
        clean_patch_metadata=clean_patch_metadata,
    )
    cache = model.prepare_canvas_cache(
        bank,
        branch_valid,
        branch_starts,
        allow_synthetic_branch_suffix=True,
    )
    cached = model.forward_prepared(cache, noisy_ids, times).branch_logits
    maximum_error = float((full - cached).abs().max())
    if not math.isfinite(maximum_error) or maximum_error > 1e-5:
        raise ValueError(
            f"quantized cached inference disagrees with full forward: {maximum_error}"
        )
    return {
        "full_vs_cached_max_abs_error": maximum_error,
        "finite": bool(torch.isfinite(cached).all()),
        "clean_atoms": clean_atoms,
        "canvas_atoms": stride,
        "branch_start": branch_start,
        "origin_phase": smoke_phase,
        "origin_stride": model.config.duo_origin_stride,
        "origin_policy": (
            "exact_non_aligned_prompt"
            if full_resolution
            else "legacy_patch_aligned_prompt"
        ),
        "patching_policy": model.config.duo_clean_patching,
        "entropy_patcher_sha256": (
            entropy_patcher.sha256 if entropy_patcher is not None else None
        ),
    }


def validation_compile_contract(
    config: ByteDiffusionConfig, device: torch.device
) -> dict[str, object]:
    """Describe the graph-shape policy used for quantized validation."""

    entropy_patching = config.duo_clean_patching == "causal_entropy_v1"
    return {
        "compiled": device.type == "cuda",
        "dynamic_shapes": entropy_patching if device.type == "cuda" else False,
        "patching_policy": config.duo_clean_patching,
        "shape_reason": (
            "ragged_causal_entropy_patch_counts"
            if entropy_patching
            else "fixed_stride_patch_counts"
        ),
    }


def prepare_validation_model(
    candidate_model: DuoModel, device: torch.device
) -> DuoModel:
    """Move and compile a model under its patching-policy shape contract."""

    candidate_model = candidate_model.to(device).eval()
    contract = validation_compile_contract(candidate_model.config, device)
    if not contract["compiled"]:
        return candidate_model
    # FlexAttention's eager fallback materializes the 4K×12K score matrix and
    # cannot fit. Entropy patch counts are data-dependent, so their deployed
    # validation graph must retain dynamic shapes.
    candidate_model.forward = torch.compile(  # type: ignore[method-assign]
        candidate_model.forward,
        dynamic=bool(contract["dynamic_shapes"]),
        fullgraph=False,
    )
    return candidate_model


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "byte_duo_checkpoint/v2":
        raise ValueError("export requires a Byte-Duo checkpoint")
    training = payload.get("training")
    if not isinstance(training, dict) or int(payload.get("steps", -1)) != int(
        training.get("steps", -2)
    ):
        raise ValueError("final export requires a completed Byte-Duo schedule")
    provenance = source_provenance()
    if payload.get("source_sha256") != provenance["sha256"]:
        raise ValueError("export code differs from Byte-Duo checkpoint provenance")
    config = model_config_from_dict(payload["model_config"])
    model = DuoModel(config, schedule_eps=float(training["schedule_eps"]))
    model.load_state_dict(payload["model"], strict=True)

    dataset_manifest = json.loads((args.data_path / "manifest.json").read_text())
    dataset_sha256 = dataset_manifest.get("payload_sha256")
    if dataset_sha256 != payload.get("dataset_payload_sha256"):
        raise ValueError("export dataset differs from Byte-Duo checkpoint")
    entropy_patcher = checkpoint_entropy_patcher_artifact(payload, args.data_path)
    entropy_patcher_model = (
        CausalEntropyPatcher.from_bytes(entropy_patcher)
        if entropy_patcher is not None
        else None
    )
    atomic_manifest, _, validation = load_data_directory(
        args.data_path,
        chunk_size=int(training.get("chunk_size", 8192)),
        recipe="blt_d",
        required_branch_bytes=int(training["canvas_length"]) * int(training["branches"]),
        branch_span_length=int(training["canvas_length"]),
        validation_chunk_limit=args.validation_rows,
        expected_payload_sha256=dataset_sha256,
    )
    paths = counted_duo_code_paths()
    code_bytes = sum(path.stat().st_size for path in paths)
    candidate = build_artifact(
        model,
        config,
        group_size=args.group_size,
        encoding_policy=args.encoding_policy,
        code_bytes=code_bytes,
        atomic_manifest=atomic_manifest,
        entropy_patcher=entropy_patcher,
    )
    quantized = DuoModel(config, schedule_eps=float(training["schedule_eps"]))
    load_artifact(quantized, candidate)
    cached_smoke = cached_inference_smoke(
        quantized.eval(), entropy_patcher_model
    )
    if not args.cpu_reference and not torch.cuda.is_available():
        raise RuntimeError("CUDA export evaluation requires mlq; use --cpu-reference in tests")
    device = torch.device("cpu" if args.cpu_reference else "cuda")
    evaluated_rows = min(args.validation_rows, len(validation))
    validation_batches = tuple(
        _validation_batches(
            validation,
            rows=args.validation_rows,
            batch_size=args.validation_batch_size,
            patch_stride=config.patch_stride,
            device=torch.device("cpu"),
        )
    )
    prepared_batches = tuple(
        PreparedDuoValidationBatch(
            batch,
            prepare_duo_validation_inputs(
                model,
                batch,
                row_ids=batch.row_ids,
                total_rows=evaluated_rows,
                canvas_length=int(training["canvas_length"]),
                branches=int(training["branches"]),
                seed=int(training["validation_seed"]),
            ),
        )
        for batch in validation_batches
    )
    model = prepare_validation_model(model, device)
    float_evaluation = asdict(
        validate_duo(
            model,
            tuple(batch.to(device) for batch in prepared_batches),
            canvas_length=int(training["canvas_length"]),
            branches=int(training["branches"]),
            seed=int(training["validation_seed"]),
            total_rows=evaluated_rows,
            compute_ar_diagnostic=True,
        )
    )
    model = model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    quantized = prepare_validation_model(quantized, device)
    evaluation = asdict(
        validate_duo(
            quantized,
            tuple(batch.to(device) for batch in prepared_batches),
            canvas_length=int(training["canvas_length"]),
            branches=int(training["branches"]),
            seed=int(training["validation_seed"]),
            total_rows=evaluated_rows,
            compute_ar_diagnostic=True,
        )
    )
    quant_delta_bits = (
        float(evaluation["conditional_canvas_nelbo_nats_per_atom"])
        - float(float_evaluation["conditional_canvas_nelbo_nats_per_atom"])
    ) / math.log(2.0)
    if quant_delta_bits > args.max_quantization_delta_bits:
        raise ValueError(
            f"quantization degraded NELBO by {quant_delta_bits:.6f} bits/atom; "
            f"limit is {args.max_quantization_delta_bits:.6f}"
        )
    evaluation["float_reference"] = float_evaluation
    evaluation["quantization_delta_bits_per_atom"] = quant_delta_bits
    evaluation["max_quantization_delta_bits_per_atom"] = (
        args.max_quantization_delta_bits
    )
    evaluation["cached_inference_smoke"] = cached_smoke
    evaluation["validation_compile_contract"] = validation_compile_contract(
        config, device
    )
    candidate_metadata, _ = parse_artifact(candidate)
    evaluation["encoding_plan"] = {
        "policy": candidate_metadata["encoding_policy"],
        "group_size": candidate_metadata["group_size"],
        "fallbacks": candidate_metadata["encoding_fallbacks"],
    }
    evaluation["validation_rows"] = evaluated_rows
    artifact_provenance = {
        "architecture": "byte_duo_uniform_state_diffusion",
        "checkpoint_schema": payload["schema"],
        "checkpoint_step": int(payload["steps"]),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "dataset_payload_sha256": dataset_sha256,
        "source_sha256": provenance["sha256"],
        "complete_code": complete_code_provenance(),
        "metric_semantics": "conditional_canvas_duo_nelbo_not_ar_bpb",
        "canvas_length": int(training["canvas_length"]),
        "patching_policy": config.duo_clean_patching,
        "entropy_patcher_sha256": (
            hashlib.sha256(entropy_patcher).hexdigest()
            if entropy_patcher is not None
            else None
        ),
    }
    artifact = build_artifact(
        model,
        config,
        group_size=args.group_size,
        encoding_policy=args.encoding_policy,
        code_bytes=code_bytes,
        atomic_manifest=atomic_manifest,
        post_quantization_metrics=evaluation,
        provenance=artifact_provenance,
        entropy_patcher=entropy_patcher,
        require_evaluation=True,
    )
    digest = write_artifact(args.output, artifact)
    metadata, _ = parse_artifact(artifact)
    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "artifact_bytes": len(artifact),
                "code_bytes": code_bytes,
                "complete_bytes": len(artifact) + code_bytes,
                "headroom_bytes": 16_000_000 - len(artifact) - code_bytes,
                "parameter_count": metadata["parameter_count"],
                "post_quantization_metrics": evaluation,
                "sha256": digest,
                "counted_code_files": tuple(
                    str(path.relative_to(REPO_ROOT)) for path in paths
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
