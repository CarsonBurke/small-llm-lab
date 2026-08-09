"""Command-line configuration for the independent OPSD trainer."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_CHECKPOINT = (
    "postraining/runs/sft6_bare_a1swap10k_e3/sft_final_model.pt"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "On-policy self-distillation with a frozen step-0 privileged "
            "teacher (Zhao et al., arXiv:2601.18734v3)."
        )
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Verified problem/solution parquet. Defaults to the source "
            "checkpoint's SFT traces, or sft_traces_v1 for a pretrained base."
        ),
    )
    parser.add_argument(
        "--reference-column",
        default="auto",
        help=(
            "Privileged reference column. 'auto' uses document when present, "
            "otherwise solution; controls may select permuted_solution."
        ),
    )
    parser.add_argument(
        "--data-manifest",
        default=None,
        help=(
            "Dataset-build manifest. Required for explicit final-answer "
            "solution and permuted-solution arms."
        ),
    )
    parser.add_argument(
        "--authorization",
        default=None,
        help=(
            "Dual frozen-gate pass artifact. Required for explicit "
            "final-answer solution and permuted-solution training arms."
        ),
    )
    parser.add_argument(
        "--allow-failed-authorization",
        action="store_true",
        help=(
            "Explicitly run an experimental final-answer arm even when its "
            "bound authorization artifact records decision=fail. The failed "
            "artifact remains required and is preserved in run provenance."
        ),
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--rollout-batch-size", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--distillation-temperature",
        type=float,
        default=None,
        help="Defaults to --temperature, matching the authors' implementation.",
    )
    parser.add_argument("--pointwise-kl-clip", type=float, default=0.05)
    parser.add_argument("--logit-chunk-tokens", type=int, default=32)
    parser.add_argument(
        "--rollout-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compile the recurrent decode step with dynamic shapes. This is "
            "the production default; use --no-rollout-compile only for "
            "debugging compiler failures."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--generation-log-samples", type=int, default=2)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help=(
            "Optionally evaluate the question-only policy on --eval-data at "
            "step 0 and this cadence. Disabled by default because live "
            "VAPO-style train accuracy grades existing rollouts without "
            "additional generation."
        ),
    )
    parser.add_argument(
        "--eval-data",
        default="postraining/data/deepmind-interpolate-easy.parquet",
        help=(
            "Verifier dataset excluded from SFT and OPSD training. The "
            "default is the fixed DeepMind interpolate-easy panel."
        ),
    )
    parser.add_argument("--eval-rows", type=int, default=144)
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--eval-batch-trajectories", type=int, default=128)
    parser.add_argument("--eval-temperature", type=float, default=1.1)
    parser.add_argument("--eval-top-p", type=float, default=0.95)
    parser.add_argument("--eval-top-k", type=int, default=20)
    parser.add_argument("--eval-think-min-tokens", type=int, default=33)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--think-tokens",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Infer from SFT provenance by default.",
    )
    parser.add_argument(
        "--answer-fence",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Infer from SFT provenance by default.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate checkpoint, data, prompt, tokenizer, and context "
            "contracts without CUDA."
        ),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    name = Path(args.name)
    if name.is_absolute() or len(name.parts) != 1 or args.name in (".", ".."):
        raise ValueError("--name must be one run-directory name")
    positive_ints = (
        "steps",
        "effective_batch_size",
        "rollout_batch_size",
        "max_completion_length",
        "max_prompt_length",
        "logit_chunk_tokens",
        "save_every",
        "eval_rows",
        "eval_samples",
        "eval_batch_trajectories",
        "eval_think_min_tokens",
    )
    for field in positive_ints:
        if getattr(args, field) < 1:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.rollout_batch_size > args.effective_batch_size:
        raise ValueError(
            "--rollout-batch-size cannot exceed --effective-batch-size"
        )
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.distillation_temperature is not None and (
        args.distillation_temperature <= 0
    ):
        raise ValueError("--distillation-temperature must be positive")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if args.top_k < 0:
        raise ValueError("--top-k must be nonnegative")
    if args.pointwise_kl_clip < 0:
        raise ValueError("--pointwise-kl-clip must be nonnegative")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.max_grad_norm <= 0:
        raise ValueError("--max-grad-norm must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be nonnegative")
    if args.generation_log_samples < 0:
        raise ValueError("--generation-log-samples must be nonnegative")
    if args.eval_every < 0:
        raise ValueError("--eval-every must be nonnegative")
    if args.eval_temperature <= 0:
        raise ValueError("--eval-temperature must be positive")
    if not 0 < args.eval_top_p <= 1:
        raise ValueError("--eval-top-p must be in (0, 1]")
    if args.eval_top_k < 0:
        raise ValueError("--eval-top-k must be nonnegative")
    if args.answer_fence and args.think_tokens is False:
        raise ValueError("--answer-fence requires --think-tokens")


def resolved_distillation_temperature(args: argparse.Namespace) -> float:
    return (
        args.temperature
        if args.distillation_temperature is None
        else args.distillation_temperature
    )
