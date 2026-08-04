#!/usr/bin/env python3
"""Evaluate order-invariant patch-set readouts on a frozen answer encoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from postraining.answer_encoder.model import DEFAULT_REWARD_TEMPERATURE
from postraining.answer_encoder.inference import (
    FrozenAnswerScorer,
    FrozenPatchSetScorer,
    load_encoder_checkpoint,
)
from postraining.answer_encoder.probes import run_behavioral_probes
from postraining.answer_encoder.train import gpt2_encode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--patch-space", choices=("projection", "backbone"), default="projection"
    )
    parser.add_argument(
        "--reward-temperature", type=float, default=DEFAULT_REWARD_TEMPERATURE
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("patch-set probing is a model workload and requires CUDA")
    model, payload = load_encoder_checkpoint(args.checkpoint, device)
    reports = {
        "cls": run_behavioral_probes(
            FrozenAnswerScorer(
                model,
                gpt2_encode,
                device=device,
                space=args.patch_space,
                reward_temperature=args.reward_temperature,
            )
        )
    }
    for name, penalize_unmatched in (
        ("cardinality_aware", True),
        ("matched_only", False),
    ):
        scorer = FrozenPatchSetScorer(
            model,
            gpt2_encode,
            device=device,
            space=args.patch_space,
            penalize_unmatched=penalize_unmatched,
            reward_temperature=args.reward_temperature,
        )
        reports[name] = run_behavioral_probes(scorer)

    result = {
        "schema": "text_lejepa_patch_set_probe/v2",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": payload.get("step"),
        "patch_space": args.patch_space,
        "reward_mapping": "exp((cosine - 1) / temperature)",
        "reward_temperature": args.reward_temperature,
        "reports": reports,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
