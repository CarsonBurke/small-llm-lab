"""Evaluate an OPSD policy question-only on the immutable DAPO gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch

from pretraining.fresh_lejepa.fresh_lejepa_train import FreshHyperparameters
from postraining.core import load_posttraining_tokenizer
from postraining.latent_eval import evaluate_latent_math
from postraining.latent_thought import LatentThoughtModel
from postraining.math_prompt import require_answer_fence_prompt_schema
from postraining.model_io import load_model
from postraining.opsd.prepare_dapo import file_sha256
from postraining.opsd.teacher_uplift import (
    arm_rows,
    atomic_json,
    summarize_attempts,
)


OPSD_POLICY_EVAL_SCHEMA = "opsd_dapo_question_policy_eval/v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--gate-data",
        default="postraining/data/opsd_dapo17k_contractlast_gate.parquet",
    )
    parser.add_argument(
        "--data-manifest",
        default="postraining/data/opsd_dapo17k_contractlast.manifest.json",
    )
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--think-min-tokens", type=int, default=33)
    parser.add_argument("--batch-trajectories", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rollout-compile", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    if min(
        args.rows,
        args.samples,
        args.max_completion_length,
        args.think_min_tokens,
        args.batch_trajectories,
    ) < 1:
        parser.error("row, sample, token, and batch counts must be positive")
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        parser.error("invalid sampling configuration")

    output = Path("postraining/runs") / args.name / "policy_eval"
    if output.exists():
        parser.error(f"refusing to overwrite existing policy eval {output}")
    checkpoint_path = Path(args.checkpoint)
    gate_path = Path(args.gate_data)
    manifest = json.loads(Path(args.data_manifest).read_text())
    if file_sha256(gate_path) != manifest.get("gate_sha256"):
        parser.error("gate parquet does not match its manifest hash")
    if args.rows != manifest.get("gate_rows"):
        parser.error("rows must equal the full immutable DAPO gate size")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sft = payload.get("sft") or {}
    require_answer_fence_prompt_schema(
        sft, answer_fence=True, source="OPSD policy-eval checkpoint"
    )
    if sft.get("traces_sha256") != manifest.get("sft_corpus_sha256"):
        parser.error("checkpoint SFT corpus does not match DAPO gate manifest")
    context_tokens = int(payload.get("train_seq_len", 1024))
    if args.max_completion_length >= context_tokens:
        parser.error("completion length must leave room for the prompt")

    rows = pq.read_table(gate_path).to_pylist()
    if len(rows) != args.rows:
        parser.error(f"gate contains {len(rows)} rows, expected {args.rows}")
    tokenizer = load_posttraining_tokenizer(
        payload["architecture"],
        FreshHyperparameters.tokenizer_path,
        think_tokens=True,
        answer_tokens=True,
    )
    device = torch.device("cuda")
    model = load_model(checkpoint_path, device, payload=payload)
    del payload
    wrapper = LatentThoughtModel(model, num_blocks=0).to(device)
    wrapper.eval()
    compiled_step = None
    if args.rollout_compile:
        compiled_step = torch.compile(
            wrapper.step_core,
            mode="max-autotune-no-cudagraphs",
            fullgraph=True,
            dynamic=True,
        )
    think_ids = (tokenizer.think_open_id, tokenizer.think_close_id)
    answer_ids = (tokenizer.answer_open_id, tokenizer.answer_close_id)
    stop_ids = tuple(
        dict.fromkeys(
            token
            for token in (tokenizer.eos_id(), tokenizer.bos_id())
            if token >= 0
        )
    )
    attempts: list[dict] = []
    metrics = evaluate_latent_math(
        wrapper,
        tokenizer,
        arm_rows(rows, "question_only"),
        args.samples,
        args.max_completion_length,
        args.max_completion_length,
        args.samples,
        args.seed,
        device,
        context_tokens - args.max_completion_length,
        batch_trajectories=args.batch_trajectories,
        compiled_step_core=compiled_step,
        captured_attempts=attempts,
        capture_problem_count=len(rows),
        capture_samples_per_problem=args.samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        pin_emit=True,
        think_fence_ids=think_ids,
        answer_fence_ids=answer_ids,
        min_think_tokens=args.think_min_tokens,
    )
    if metrics["compile_fallback"]:
        raise RuntimeError(
            "policy eval compiled rollout fell back; rerun with "
            "--no-rollout-compile"
        )
    metrics.update(
        summarize_attempts(attempts, rows, tokenizer, stop_ids, answer_ids)
    )
    output.mkdir(parents=True)
    atomic_json(
        {"schema": OPSD_POLICY_EVAL_SCHEMA, "attempts": attempts},
        output / "attempts.json",
    )
    result = {
        "schema": OPSD_POLICY_EVAL_SCHEMA,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "gate_data": str(gate_path),
        "gate_sha256": manifest["gate_sha256"],
        "metrics": metrics,
        "args": vars(args),
    }
    atomic_json(result, output / "results.json")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
