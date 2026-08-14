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

from pretraining.byte_diffusion.duo_model import DuoModel
from pretraining.byte_diffusion.export import build_artifact, load_artifact, parse_artifact, write_artifact
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


@torch.no_grad()
def cached_inference_smoke(model: DuoModel) -> dict[str, object]:
    """Verify the artifact's deployed cached path against its full forward."""

    stride = model.config.patch_stride
    clean_length = 2 * stride
    clean_ids = torch.arange(clean_length, dtype=torch.long).remainder(256)[None]
    clean_valid = torch.ones_like(clean_ids, dtype=torch.bool)
    document_ids = torch.zeros_like(clean_ids)
    positions = torch.arange(clean_length, dtype=torch.long)[None]
    noisy_ids = torch.arange(stride, dtype=torch.long)[None, None]
    branch_valid = torch.ones_like(noisy_ids, dtype=torch.bool)
    branch_starts = torch.full((1, 1), clean_length - stride, dtype=torch.long)
    times = torch.full((1, 1), 0.5)
    full = model(
        clean_ids,
        clean_valid,
        document_ids,
        positions,
        noisy_ids,
        branch_valid,
        branch_starts,
        times,
        allow_synthetic_branch_suffix=True,
    ).branch_logits
    bank = model.prepare_clean_bank(clean_ids, clean_valid, document_ids, positions)
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
        "clean_atoms": clean_length,
        "canvas_atoms": stride,
    }


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
    )
    quantized = DuoModel(config, schedule_eps=float(training["schedule_eps"]))
    load_artifact(quantized, candidate)
    cached_smoke = cached_inference_smoke(quantized.eval())
    if not args.cpu_reference and not torch.cuda.is_available():
        raise RuntimeError("CUDA export evaluation requires mlq; use --cpu-reference in tests")
    device = torch.device("cpu" if args.cpu_reference else "cuda")
    def prepare_for_validation(candidate_model: DuoModel) -> DuoModel:
        candidate_model = candidate_model.to(device).eval()
        if device.type != "cuda":
            return candidate_model
        # FlexAttention's eager fallback materializes the 4K×12K score matrix
        # and cannot fit. The shipped inference/training path is compiled; the
        # post-quantization check must exercise that same fused path.
        candidate_model.forward = torch.compile(  # type: ignore[method-assign]
            candidate_model.forward, dynamic=False, fullgraph=False
        )
        return candidate_model
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
    model = prepare_for_validation(model)
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
    quantized = prepare_for_validation(quantized)
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
