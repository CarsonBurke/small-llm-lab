"""Measure a checkpoint's best-case DAPO-Math verifier hit rate.

Fresh-combiner hidden-carry rollouts (identity at init, so exactly the
pretrained token policy) over many prompts and samples; reports verifier
hits, within-group reward variance, and the extracted-answer distribution.
This is the cheap go/no-go probe before an RL run: zero within-group
variance means a zero policy gradient, so DAPO RL cannot start from that
checkpoint.

    python3 -m postraining.dapo_hit_rate_probe \
        --checkpoint ablation_results/<run>/pretraining_checkpoint.pt
"""

from __future__ import annotations

import argparse
import collections

import torch

import train_gpt as baseline  # noqa: F401  (import order: patches must load first)
from fresh_lejepa_train import FreshHyperparameters
from postraining.core import (
    POSTTRAIN_CONTEXT_TOKENS,
    POSTTRAIN_PROMPT_TOKENS,
    POSTTRAIN_RESPONSE_TOKENS,
    answer_style,
    encode_prompt,
    load_posttraining_tokenizer,
    load_unique_math_rows,
    validate_posttraining_context_budget,
)
from postraining.latent_eval import verify_terminated_answer
from postraining.latent_rollout import (
    emitted_token_rows,
    rollout_continuations,
    trim_stream,
)
from postraining.latent_thought import LatentThoughtModel
from postraining.math_prompt import require_answer_fence_prompt_schema
from postraining.model_io import load_model
from postraining.train_latent_vapo import rewrite_prompts_for_answer_fence
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
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    device = torch.device("cuda")
    checkpoint_payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    backbone = load_model(args.checkpoint, device, payload=checkpoint_payload)
    sft_provenance = (
        (checkpoint_payload.get("sft") or {})
        if isinstance(checkpoint_payload, dict)
        else {}
    )
    sft_args = sft_provenance.get("args") or {}
    del checkpoint_payload
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
    validate_posttraining_context_budget(
        args.prompt_tokens, args.max_new_tokens, context_tokens
    )
    backbone.eval()
    # A fresh combiner is an exact identity, so this rollout is the pretrained
    # token policy with the hidden carry plumbed but inert.
    wrapper = LatentThoughtModel(backbone).to(device).eval()

    # An SFT base checkpoint may be fence-trained; grading it through a
    # fence-less tokenizer decodes the fence ids away and reads near-zero.
    probe_answer_fence = bool(sft_args.get("answer_fence"))
    require_answer_fence_prompt_schema(
        sft_provenance,
        answer_fence=probe_answer_fence,
        source=args.checkpoint,
    )
    tokenizer = load_posttraining_tokenizer(
        backbone.architecture,
        FreshHyperparameters.tokenizer_path,
        think_tokens=bool(sft_args.get("think_tokens")),
        answer_tokens=probe_answer_fence,
    )
    answer_fence_ids = (
        (tokenizer.answer_open_id, tokenizer.answer_close_id)
        if probe_answer_fence
        else None
    )
    # dict.fromkeys dedupes while keeping order: GPT-2's single <|endoftext|>
    # token reports as both EOS and BOS.
    stop_ids = tuple(dict.fromkeys(
        t for t in (tokenizer.eos_id(), tokenizer.bos_id()) if t >= 0
    ))
    rows = load_unique_math_rows(args.math_data)[: args.prompts]
    if probe_answer_fence:
        # A fence-trained policy probed under the plain-text Answer:
        # instruction is off-distribution; frame the task as SFT did.
        rows = rewrite_prompts_for_answer_fence(rows)

    torch.manual_seed(args.seed)
    hits = 0
    total = 0
    groups_with_variance = 0
    answer_lines = 0
    extractions: collections.Counter[str] = collections.Counter()
    for index, row in enumerate(rows):
        prompt_ids = torch.tensor(
            encode_prompt(tokenizer, prompt_text(row), args.prompt_tokens),
            dtype=torch.long,
            device=device,
        )[None]
        truth = row["reward_model"]["ground_truth"]
        with torch.no_grad():
            batch = trim_stream(
                rollout_continuations(
                    wrapper,
                    prompt_ids.expand(args.samples, -1),
                    args.max_new_tokens,
                    args.max_new_tokens,
                    args.temperature,
                    args.top_p,
                    stop_ids=stop_ids or None,
                )
            )
        group_hits = 0
        for emitted in emitted_token_rows(batch):
            correct, prediction = verify_terminated_answer(
                emitted, truth, tokenizer, stop_ids, answer_style(row),
                answer_fence_ids=answer_fence_ids,
            )
            extractions[str(prediction)] += 1
            answer_lines += int(
                str(prediction) not in {"[INVALID]", "[UNTERMINATED]"}
            )
            hit = int(correct)
            group_hits += hit
            total += 1
        hits += group_hits
        if 0 < group_hits < args.samples:
            groups_with_variance += 1
        if (index + 1) % 16 == 0:
            print(
                f"{index + 1}/{len(rows)} prompts: {hits}/{total} hits, "
                f"{groups_with_variance} groups with variance",
                flush=True,
            )

    print(f"\nhits: {hits}/{total} ({hits / max(total, 1):.4%})")
    print(f"Answer-line compliance: {answer_lines}/{total} "
          f"({answer_lines / max(total, 1):.4%})")
    print(f"groups with within-group variance: {groups_with_variance}/{len(rows)}")
    print(f"top extracted answers: {extractions.most_common(15)}")


if __name__ == "__main__":
    main()
