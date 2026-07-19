"""Measure a checkpoint's best-case DAPO-Math verifier hit rate.

Emit-only rollouts (gate pinned to EMIT, no latent thinking) over many
prompts and samples; reports verifier hits, within-group reward variance,
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
    POSTTRAIN_PROMPT_TOKENS,
    POSTTRAIN_RESPONSE_TOKENS,
    POSTTRAIN_STREAM_TOKENS,
    encode_prompt,
    load_unique_math_rows,
    validate_posttraining_context_budget,
    verify_answer,
)
from postraining.latent_rollout import (
    emitted_token_rows,
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
    parser.add_argument(
        "--prompt-tokens", type=int, default=POSTTRAIN_PROMPT_TOKENS
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=POSTTRAIN_RESPONSE_TOKENS
    )
    parser.add_argument(
        "--max-stream-steps", type=int, default=POSTTRAIN_STREAM_TOKENS
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    validate_posttraining_context_budget(args.prompt_tokens, args.max_stream_steps)

    device = torch.device("cuda")
    backbone = load_model(args.checkpoint, device)
    backbone.eval()
    wrapper = LatentThoughtModel(backbone).to(device).eval()
    with torch.no_grad():
        # Zero gate weight + saturated bias: EMIT with probability ~1, so
        # this measures the belief renderer without latent-thought noise.
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
                    args.max_stream_steps,
                    args.temperature,
                    args.top_p,
                    stop_ids=stop_ids or None,
                )
            )
        group_hits = 0
        for emitted in emitted_token_rows(batch):
            cut = next((i for i, t in enumerate(emitted) if t in stop_ids), None)
            if cut is not None:
                emitted = emitted[: cut + 1]
            correct, prediction = verify_answer(tokenizer.decode(emitted), truth)
            extractions[str(prediction)] += 1
            answer_lines += int(str(prediction) != "[INVALID]")
            group_hits += int(correct)
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
