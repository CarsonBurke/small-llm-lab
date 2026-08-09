#!/usr/bin/env python3
"""Measure the complete default byte-diffusion artifact without a checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.byte_diffusion.config import ByteDiffusionConfig
from pretraining.byte_diffusion.data import AtomicIdManifest
from pretraining.byte_diffusion.export import build_artifact, parse_artifact
from pretraining.byte_diffusion.model import ByteDiffusionModel
from pretraining.byte_diffusion.training import CHECKPOINT_SCHEMA
from scripts.export_byte_diffusion import counted_code_paths


def main() -> None:
    config = ByteDiffusionConfig()
    model = ByteDiffusionModel(config)
    paths = counted_code_paths()
    code_bytes = sum(path.stat().st_size for path in paths)
    artifact = build_artifact(
        model,
        config,
        code_bytes=code_bytes,
        atomic_manifest=AtomicIdManifest.reference(),
        post_quantization_metrics={
            "ar_loss": 0.0,
            "bpb": 0.0,
            "atomic_bpb": 0.0,
            "diffusion_loss": 0.0,
            "ar_targets": 0,
            "literal_bytes": 0,
            "special_targets": 0,
            "diffusion_targets": 0,
            "diffusion_role_nll": (0.0,) * 6,
            "diffusion_role_counts": (0,) * 6,
            "audit_only_untrained": 1,
        },
        provenance={
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "checkpoint_step": 2000,
            "checkpoint_sha256": "0" * 64,
            "training_dataset_sha256": "0" * 64,
            "validation_dataset_sha256": "0" * 64,
        },
    )
    metadata, _ = parse_artifact(artifact)
    print(
        json.dumps(
            {
                "artifact_bytes": len(artifact),
                "code_bytes": code_bytes,
                "complete_bytes": len(artifact) + code_bytes,
                "headroom_bytes": 16_000_000 - len(artifact) - code_bytes,
                "parameter_count": metadata["parameter_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
