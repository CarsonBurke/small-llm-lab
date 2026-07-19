"""AIME 2024 avg@32 evaluation matching VAPO/DAPO sampling settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm
import torch

from postraining.core import (
    POSTTRAIN_RESPONSE_TOKENS,
    load_unique_math_rows,
    verify_answer,
)
from postraining.model_io import load_model
from postraining.train_vapo import generate_group, prompt_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-file", default="postraining/data/aime-2024.parquet")
    parser.add_argument("--tokenizer", default="data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=POSTTRAIN_RESPONSE_TOKENS)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", default="postraining/runs/fresh_lejepa_vapo/aime_metrics.jsonl")
    parser.add_argument(
        "--min-accuracy", type=float, default=None,
        help="exit 2 below this accuracy (pre-RL signal gate for job chains)",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    model = load_model(args.checkpoint, device)
    model.eval()
    if hasattr(model, "fold_input_projector_for_inference"):
        model.fold_input_projector_for_inference()
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    rows = load_unique_math_rows(args.test_file)
    results = []
    transcripts = []
    for problem_index, row in enumerate(rows):
        prompt = prompt_text(row)
        _, responses, _, _ = generate_group(
            model, tokenizer, prompt, args.samples, args.max_tokens,
            args.temperature, args.top_p, capture_stats=False,
        )
        ground_truth = row["reward_model"]["ground_truth"]
        for sample_index, response in enumerate(responses):
            eos = tokenizer.eos_id()
            if eos >= 0 and (response == eos).any():
                response = response[: int((response == eos).nonzero()[0]) + 1]
            text = tokenizer.decode(response.tolist())
            correct, prediction = verify_answer(text, ground_truth, "aime")
            results.append({"correct": correct, "prediction": prediction})
            transcripts.append({
                "step": args.step,
                "problem": problem_index,
                "sample": sample_index,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "text": text,
                "prediction": prediction,
                "correct": correct,
            })
    accuracy = sum(result["correct"] for result in results) / len(results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": args.step, "accuracy": accuracy, "samples": len(results)}) + "\n")
    # Full per-sample record — the aggregate alone can't answer "what did it
    # actually say", and generations are otherwise discarded.
    transcript_path = output.with_name(output.stem + f"_transcripts_step{args.step}.jsonl")
    with transcript_path.open("w", encoding="utf-8") as handle:
        for record in transcripts:
            handle.write(json.dumps(record) + "\n")
    print(f"step:{args.step} acc/mean@32:{accuracy:.6f} samples:{len(results)}")
    if args.min_accuracy is not None and accuracy < args.min_accuracy:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
