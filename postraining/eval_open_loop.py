"""Open-loop k-step BPB: how fast latent imagination degrades with depth.

The transition head is a predictive world model over projected token latents.
This eval alternates ``ground`` teacher-forced token inputs with ``imagine``
imagined steps — the model's own predicted next-token latent (mean, or a
transition sample) fed back through the adapter — and buckets next-token loss
by imagination depth d, the number of consecutive imagined inputs consumed.
Targets are always the true validation tokens, so bpb at depth d answers:
after imagining d steps, how well does the model still predict the actual
text?  Depth 0 is the closed-loop reference under the identical stepwise
protocol, and bpb uses the challenge's exact byte accounting.

    python3 -m postraining.eval_open_loop \
        --checkpoint ablation_results/<run>/pretraining_checkpoint.pt \
        [--wrapper-checkpoint postraining/runs/<name>/latent_vapo_checkpoint.pt] \
        [--ground 8 --imagine 8 --mode mean]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

import train_gpt as baseline
from fresh_lejepa_train import FreshHyperparameters
from postraining.latent_thought import (
    LatentThoughtModel,
    validate_renderer_checkpoint,
)
from postraining.model_io import load_model


@torch.no_grad()
def open_loop_depth_metrics(
    wrapper: LatentThoughtModel,
    x: Tensor,
    y: Tensor,
    ground: int,
    imagine: int,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    mode: str = "mean",
    generator: torch.Generator | None = None,
) -> list[dict[str, float]]:
    """Per-depth loss/bpb over a (batch, seq) validation slab.

    Position t is grounded when ``t % (ground + imagine) < ground`` (the
    sequence therefore starts grounded); imagined positions feed the previous
    step's predicted latent through ``thought_input``.  Byte counts follow
    ``eval_val`` and depend only on the reference text, never on the inputs
    the model consumed.
    """
    if ground < 1:
        raise ValueError("at least one grounded position per period is required")
    if imagine < 0:
        raise ValueError("imagine must be non-negative")
    if mode not in ("mean", "sample"):
        raise ValueError(f"unknown imagination mode {mode!r}")
    device = x.device
    batch, seq_len = x.shape
    period = ground + imagine
    loss_sum = torch.zeros(imagine + 1, dtype=torch.float64, device=device)
    token_count = torch.zeros_like(loss_sum)
    byte_count = torch.zeros_like(loss_sum)

    def score(position: int, depth: int, logits: Tensor) -> None:
        targets = y[:, position]
        ce = F.cross_entropy(logits.float(), targets, reduction="none")
        token_bytes = base_bytes_lut[targets].to(torch.int16) + (
            has_leading_space_lut[targets] & ~is_boundary_token_lut[x[:, position]]
        ).to(torch.int16)
        loss_sum[depth] += ce.to(torch.float64).sum()
        token_count[depth] += float(batch)
        byte_count[depth] += token_bytes.to(torch.float64).sum()

    caches = wrapper.make_generation_cache(batch, seq_len, device)
    output = wrapper.token_step(x[:, 0], caches, 0)
    caches = output.caches
    score(0, 0, output.logits)
    depth = 0
    for position in range(1, seq_len):
        if position % period < ground:
            depth = 0
            next_input = wrapper.embed_tokens(x[:, position][:, None])
        else:
            depth += 1
            if mode == "sample":
                thought, _ = wrapper.transition.sample(
                    output.predicted, generator=generator
                )
            else:
                thought = output.predicted.float()
            next_input = wrapper.thought_input(thought)
        output = wrapper.step(next_input, caches, position)
        caches = output.caches
        score(position, depth, output.logits)

    metrics = []
    for d in range(imagine + 1):
        if float(token_count[d]) == 0.0:
            continue
        loss = float(loss_sum[d] / token_count[d])
        metrics.append(
            {
                "depth": d,
                "loss": loss,
                "bpb": float(loss_sum[d] / math.log(2.0) / byte_count[d]),
                "tokens": int(token_count[d]),
            }
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wrapper-checkpoint", default=None)
    parser.add_argument("--ground", type=int, default=8)
    parser.add_argument("--imagine", type=int, default=8)
    parser.add_argument("--sequences", type=int, default=64)
    parser.add_argument("--mode", choices=("mean", "sample"), default="mean")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device("cuda")
    backbone = load_model(args.checkpoint, device)
    backbone.eval()
    wrapper = LatentThoughtModel(backbone).to(device)
    if args.wrapper_checkpoint:
        payload = torch.load(args.wrapper_checkpoint, map_location="cpu", weights_only=False)
        validate_renderer_checkpoint(payload, args.wrapper_checkpoint)
        wrapper.load_state_dict(payload["model"], strict=True)
    wrapper.eval()

    import sentencepiece as spm

    tokenizer = spm.SentencePieceProcessor(model_file=FreshHyperparameters.tokenizer_path)
    luts = baseline.build_sentencepiece_luts(tokenizer, FreshHyperparameters.vocab_size, device)
    seq_len = FreshHyperparameters.train_seq_len
    val_tokens = baseline.load_validation_tokens(FreshHyperparameters.val_files, seq_len)
    raw = val_tokens[: args.sequences * seq_len + 1].to(device=device, dtype=torch.int64)
    x = raw[:-1].reshape(-1, seq_len)
    y = raw[1:].reshape(-1, seq_len)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    # Plain fp32, matching the RL rollout/replay regime (the stepwise cache
    # path does not support autocast: fp32 caches reject bf16 values).
    metrics = open_loop_depth_metrics(
        wrapper, x, y, args.ground, args.imagine, *luts,
        mode=args.mode, generator=generator,
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "wrapper_checkpoint": args.wrapper_checkpoint and str(args.wrapper_checkpoint),
        "ground": args.ground,
        "imagine": args.imagine,
        "mode": args.mode,
        "sequences": int(x.size(0)),
        "per_depth": metrics,
    }
    print(json.dumps(report, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
