"""Re-evaluate an immutable SFT checkpoint on its held-out sampling gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from argparse import Namespace
from pathlib import Path

import torch

from pretraining.fresh_lejepa.fresh_lejepa_train import FreshHyperparameters
from postraining.core import load_posttraining_tokenizer
from postraining.math_prompt import require_answer_fence_prompt_schema
from postraining.model_io import load_model
from postraining.sft_trace_train import (
    INSTRUCTION_SUFFIX_ANSWER,
    load_documents,
    run_sampling_gate,
    split_holdout,
)


SFT_GATE_EVAL_SCHEMA = "sft_sampling_gate_eval/v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--holdout-problems", type=int, default=256)
    parser.add_argument("--gate-prompts", type=int, default=128)
    parser.add_argument("--gate-samples", type=int, default=8)
    parser.add_argument("--gate-max-new-tokens", type=int, default=768)
    parser.add_argument("--gate-prompt-tokens", type=int, default=512)
    parser.add_argument("--gate-think-min-tokens", type=int, default=33)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if min(
        args.holdout_problems,
        args.gate_prompts,
        args.gate_samples,
        args.gate_max_new_tokens,
        args.gate_prompt_tokens,
        args.gate_think_min_tokens,
    ) < 1:
        parser.error("gate counts and token budgets must be positive")
    output = Path("postraining/runs") / args.name
    if output.exists():
        parser.error(f"refusing to overwrite existing gate output {output}")

    checkpoint_path = Path(args.checkpoint)
    traces_path = Path(args.traces)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sft = payload.get("sft") or {}
    require_answer_fence_prompt_schema(
        sft, answer_fence=True, source="SFT sampling-gate checkpoint"
    )
    traces_sha256 = file_sha256(traces_path)
    if sft.get("traces_sha256") != traces_sha256:
        parser.error("checkpoint was trained on different trace bytes")
    documents = load_documents(traces_path)
    _, _, panel = split_holdout(documents, args.holdout_problems)
    device = torch.device("cuda")
    backbone = load_model(checkpoint_path, device, payload=payload)
    tokenizer = load_posttraining_tokenizer(
        payload["architecture"],
        FreshHyperparameters.tokenizer_path,
        think_tokens=True,
        answer_tokens=True,
    )
    gate_args = Namespace(
        gate_prompts=args.gate_prompts,
        gate_samples=args.gate_samples,
        gate_max_new_tokens=args.gate_max_new_tokens,
        gate_prompt_tokens=args.gate_prompt_tokens,
        gate_think_min_tokens=args.gate_think_min_tokens,
        seed=args.seed,
    )
    gate, captured = run_sampling_gate(
        backbone,
        tokenizer,
        panel,
        gate_args,
        device,
        instruction_suffix=INSTRUCTION_SUFFIX_ANSWER,
    )
    output.mkdir(parents=True)
    result = {
        "schema": SFT_GATE_EVAL_SCHEMA,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "traces": str(traces_path),
        "traces_sha256": traces_sha256,
        "gate": gate,
        "args": vars(args),
    }
    temporary = output / f"result.json.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output / "result.json")
    (output / "transcripts.json").write_text(
        json.dumps(captured, indent=2) + "\n"
    )
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
