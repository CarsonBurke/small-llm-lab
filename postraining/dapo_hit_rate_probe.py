"""Measure a checkpoint's best-case DAPO-Math verifier hit rate.

No-optional-think rollouts (gate pinned to EMIT, with exactly half of every
prompt group's members forced to start with a latent thought) over many prompts
and samples; reports verifier hits, within-group reward variance,
and the extracted-answer distribution.  This is the cheap go/no-go probe
before an RL run: zero within-group variance means a zero policy gradient,
so DAPO RL cannot start from that checkpoint.

    python3 -m postraining.dapo_hit_rate_probe \
        --checkpoint ablation_results/<run>/pretraining_checkpoint.pt
"""

from __future__ import annotations

import argparse
import collections

import torch

import train_gpt as baseline  # noqa: F401  (import order: patches must load first)
import sentencepiece as spm
from fresh_lejepa_train import FreshHyperparameters
from postraining.core import (
    POSTTRAIN_CONTEXT_TOKENS,
    POSTTRAIN_PROMPT_TOKENS,
    POSTTRAIN_RESPONSE_TOKENS,
    POSTTRAIN_STREAM_TOKENS,
    answer_style,
    encode_prompt,
    load_unique_math_rows,
    validate_posttraining_context_budget,
)
from postraining.latent_eval import verify_terminated_answer
from postraining.latent_rollout import (
    emitted_token_rows,
    half_forced_group_members,
    rollout_continuations,
    trim_stream,
)
from postraining.latent_thought import LatentThoughtModel
from postraining.model_io import load_model
from postraining.train_vapo import prompt_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--math-data", default="postraining/data/dapo-math-17k.parquet")
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument("--samples", type=int, default=16)
    # None derives the backbone defaults after the checkpoint loads:
    # fresh PoPE 1024/1024/4096; nano prompt 512 with the response scaled to
    # its recorded pretraining window (256 at seq 1024, 1024 at seq >= 2560 —
    # nano's half-truncate RoPE has no extrapolation).
    parser.add_argument("--prompt-tokens", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-stream-steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.samples < 2 or args.samples % 2:
        parser.error("--samples must be even for the 50/50 forced split")

    device = torch.device("cuda")
    backbone = load_model(args.checkpoint, device)
    is_nano = backbone.architecture.startswith("nanogpt_mini")
    context_tokens = (
        getattr(backbone, "train_context_tokens", 1024)
        if is_nano
        else POSTTRAIN_CONTEXT_TOKENS
    )
    if args.prompt_tokens is None:
        args.prompt_tokens = 512 if is_nano else POSTTRAIN_PROMPT_TOKENS
    if args.max_new_tokens is None:
        args.max_new_tokens = (
            min(1024, (context_tokens - args.prompt_tokens) // 2)
            if is_nano
            else POSTTRAIN_RESPONSE_TOKENS
        )
    if args.max_stream_steps is None:
        args.max_stream_steps = min(
            4 * args.max_new_tokens if is_nano else POSTTRAIN_STREAM_TOKENS,
            context_tokens - args.prompt_tokens,
        )
    validate_posttraining_context_budget(
        args.prompt_tokens, args.max_stream_steps, context_tokens
    )
    backbone.eval()
    wrapper = LatentThoughtModel(backbone).to(device).eval()
    with torch.no_grad():
        # Zero gate weight + saturated bias: EMIT with probability ~1 except
        # for the forced half-members. This isolates the prefix intervention
        # from any additional gate-selected thinking.
        wrapper.gate.head.weight.zero_()
        wrapper.gate.head.bias.fill_(30.0)

    tokenizer = spm.SentencePieceProcessor(
        model_file=FreshHyperparameters.tokenizer_path
    )
    stop_ids = tuple(t for t in (tokenizer.eos_id(), tokenizer.bos_id()) if t >= 0)
    rows = load_unique_math_rows(args.math_data)[: args.prompts]

    torch.manual_seed(args.seed)
    hits = 0
    total = 0
    groups_with_variance = 0
    forced_hits = 0
    unforced_hits = 0
    forced_groups_with_variance = 0
    unforced_groups_with_variance = 0
    answer_lines = 0
    extractions: collections.Counter[str] = collections.Counter()
    for index, row in enumerate(rows):
        prompt_ids = torch.tensor(
            encode_prompt(tokenizer, prompt_text(row), args.prompt_tokens),
            dtype=torch.long,
            device=device,
        )[None]
        truth = row["reward_model"]["ground_truth"]
        forced_members = half_forced_group_members(1, args.samples, device)
        with torch.no_grad():
            batch = trim_stream(
                rollout_continuations(
                    wrapper,
                    prompt_ids.expand(args.samples, -1),
                    args.max_new_tokens,
                    args.max_stream_steps,
                    args.temperature,
                    args.top_p,
                    stop_ids=stop_ids or None,
                    force_initial_think=forced_members,
                )
            )
        group_hits = 0
        forced_group_hits = 0
        unforced_group_hits = 0
        forced_member_flags = forced_members.tolist()
        for member, emitted in enumerate(emitted_token_rows(batch)):
            correct, prediction = verify_terminated_answer(
                emitted, truth, tokenizer, stop_ids, answer_style(row)
            )
            extractions[str(prediction)] += 1
            answer_lines += int(
                str(prediction) not in {"[INVALID]", "[UNTERMINATED]"}
            )
            hit = int(correct)
            group_hits += hit
            if forced_member_flags[member]:
                forced_group_hits += hit
            else:
                unforced_group_hits += hit
            total += 1
        hits += group_hits
        forced_hits += forced_group_hits
        unforced_hits += unforced_group_hits
        if 0 < group_hits < args.samples:
            groups_with_variance += 1
        cohort_size = args.samples // 2
        if 0 < forced_group_hits < cohort_size:
            forced_groups_with_variance += 1
        if 0 < unforced_group_hits < cohort_size:
            unforced_groups_with_variance += 1
        if (index + 1) % 16 == 0:
            print(
                f"{index + 1}/{len(rows)} prompts: {hits}/{total} hits, "
                f"{groups_with_variance} groups with variance",
                flush=True,
            )

    print(f"\nhits: {hits}/{total} ({hits / max(total, 1):.4%})")
    cohort_total = total // 2
    print(
        f"forced-initial hits: {forced_hits}/{cohort_total} "
        f"({forced_hits / max(cohort_total, 1):.4%})"
    )
    print(
        f"unforced-initial hits: {unforced_hits}/{cohort_total} "
        f"({unforced_hits / max(cohort_total, 1):.4%})"
    )
    print(f"Answer-line compliance: {answer_lines}/{total} "
          f"({answer_lines / max(total, 1):.4%})")
    print(f"groups with within-group variance: {groups_with_variance}/{len(rows)}")
    print(
        "forced cohort groups with variance: "
        f"{forced_groups_with_variance}/{len(rows)}"
    )
    print(
        "unforced cohort groups with variance: "
        f"{unforced_groups_with_variance}/{len(rows)}"
    )
    print(f"top extracted answers: {extractions.most_common(15)}")


if __name__ == "__main__":
    main()
