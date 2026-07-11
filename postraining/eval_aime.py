"""AIME 2024 avg@32 evaluation matching VAPO/DAPO sampling settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm
import torch

from postraining.core import load_unique_math_rows, verify_answer
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
    parser.add_argument("--max-tokens", type=int, default=20480)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", default="postraining/runs/fresh_lejepa_vapo/aime_metrics.jsonl")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    model = load_model(args.checkpoint, device)
    model.eval()
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    rows = load_unique_math_rows(args.test_file)
    results = []
    for row in rows:
        _, responses = generate_group(
            model, tokenizer, prompt_text(row), args.samples, args.max_tokens,
            args.temperature, args.top_p,
        )
        ground_truth = row["reward_model"]["ground_truth"]
        for response in responses:
            eos = tokenizer.eos_id()
            if eos >= 0 and (response == eos).any():
                response = response[: int((response == eos).nonzero()[0]) + 1]
            text = tokenizer.decode(response.tolist())
            correct, prediction = verify_answer(text, ground_truth)
            results.append({"correct": correct, "prediction": prediction})
    accuracy = sum(result["correct"] for result in results) / len(results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": args.step, "accuracy": accuracy, "samples": len(results)}) + "\n")
    print(f"step:{args.step} acc/mean@32:{accuracy:.6f} samples:{len(results)}")


if __name__ == "__main__":
    main()
