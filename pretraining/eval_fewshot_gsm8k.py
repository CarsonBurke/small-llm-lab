"""Few-shot GSM8K exact-match for a pretrained (pre-SFT) source checkpoint.

Base-model GSM8K is a standard k-shot generative benchmark: prompt with k
worked exemplars, greedily continue the test question, and exact-match the
number after ``####``. Generation uses the same heterogeneous dense-KV/KDA
state caches as post-training rollouts: one batched prefill and one cached
model forward per generated token.

The GSM8K *test* split is genuinely held out of pretraining: `data/problem_registry/v1`
reserves it under the ``eval`` split, which outranks ``pretrain``, behind a
13-gram index and 344,309 excluded keys. The *train* split is not equally
protected -- ``openmath_instruct`` carries a band that is overwhelmingly
GSM8K-train -- and the exemplars here are drawn from that train split, so the
few-shot demonstrations may be familiar to the model. That biases scores up,
not down.

Scope: source-tokenizer nanoGPT checkpoints. The byte-diffusion family has a
separate evaluator because its generation units, UTF-8 validity checks, and
native-action accounting differ.

Read-only with respect to data and checkpoints, and refuses to overwrite its
own output. Run through mlq: it executes a model on the GPU.

    mlq submit --name gsm8k_anneal1k --cwd "$PWD" --max-parallel-runs 1 -- \
      .venv/bin/python -m pretraining.eval_fewshot_gsm8k \
        --checkpoint logs/k3_v8_armC_kda8_cosine_nextlat_nope_anneal1k_final_model.pt \
        --shots 1,5 --seeds 0,1,2 --output ablation_results/gsm8k_anneal1k.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from pretraining.byte_accounting import load_bound_tokenizer
from postraining.latent_thought import LatentThoughtModel
from postraining.model_io import load_model

# The canonical OpenAI release, as cached by `datasets`. The repository's own
# `gsm8k_test_questions.parquet` carries questions without gold answers, so it
# cannot score anything.
GSM8K_ROOT = Path(
    "~/.cache/huggingface/hub/datasets--openai--gsm8k/snapshots/"
    "740312add88f781978c0658806c59bc2815b9866/main"
).expanduser()
TRAIN_PARQUET = GSM8K_ROOT / "train-00000-of-00001.parquet"
TEST_PARQUET = GSM8K_ROOT / "test-00000-of-00001.parquet"

GOLD_DELIMITER = "####"
CALCULATOR_SPAN = re.compile(r"<<[^>]*>>")

# Verbatim from `scripts/build_math_mix_dataset.py`, which rendered 20% of the
# `openmath_instruct` documents this way during pretraining.
DAPO_PREAMBLE = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: $Answer (without quotes) where "
    "$Answer is the answer to the problem."
)
DAPO_REMINDER = 'Remember to put your answer on its own line after "Answer:".'


def answer_pattern(delimiter: str) -> re.Pattern:
    """Match a final answer introduced by ``delimiter``.

    Digit grouping must be well-formed 3-digit runs. A bare ``\\d[\\d,]*``
    turns the model's ``1,8`` -- two numbers run together -- into 18, which
    scores as correct against a gold of 18.
    """

    return re.compile(
        rf"{re.escape(delimiter)}\s*(-?(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?)"
    )


GOLD_PATTERN = answer_pattern(GOLD_DELIMITER)


@dataclass(frozen=True)
class PromptFormat:
    """How a question is posed, and how the answer is recognized.

    Format is the experimental variable, not a detail. The pretraining corpus
    contains no ``Question:``/``####`` documents at all: `openmath_instruct`,
    the largest math source at 8% of tokens and the one carrying GSM8K-derived
    word problems, renders as ``{problem}\\n{solution}\\nAnswer: {answer}``,
    one fifth of it inside the DAPO template. Scoring a base model outside the
    format it was trained on measures format transfer, not capability.
    """

    name: str
    delimiter: str
    stops: tuple[str, ...]

    def solution(self, answer: str, *, strip_calculator: bool) -> str:
        """Rewrite a GSM8K reference solution into this format's answer style."""

        body = CALCULATOR_SPAN.sub("", answer) if strip_calculator else answer
        reasoning, _, final = body.partition(GOLD_DELIMITER)
        if self.delimiter == GOLD_DELIMITER:
            return body
        return f"{reasoning.strip()}\n{self.delimiter} {final.strip()}"

    def exemplar(self, question: str, solution: str) -> str:
        if self.name == "harness":
            return f"Question: {question}\nAnswer: {solution}"
        if self.name == "bare":
            return f"{question}\n{solution}"
        return f"{DAPO_PREAMBLE}\n\n{question}\n\n{DAPO_REMINDER}\n{solution}"

    def query(self, question: str) -> str:
        if self.name == "harness":
            return f"Question: {question}\nAnswer:"
        if self.name == "bare":
            return f"{question}\n"
        return f"{DAPO_PREAMBLE}\n\n{question}\n\n{DAPO_REMINDER}\n"


PROMPT_FORMATS = {
    # The lm-evaluation-harness convention, and the reason to keep it: it is
    # comparable to published base-model numbers. It matches nothing in this
    # corpus.
    "harness": PromptFormat(
        "harness", GOLD_DELIMITER, ("\nQuestion:", "\nAnswer:", "\n\n")
    ),
    "bare": PromptFormat("bare", "Answer:", ("\n\n",)),
    "dapo": PromptFormat("dapo", "Answer:", ("\n\n",)),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_number(text: str) -> str | None:
    """Canonicalize an answer so 1,000 and 1000.0 compare equal.

    Integers are canonicalized exactly. Going through float first would make
    12345678901234567 and ...568 compare equal above 2**53, and would turn a
    long repetition-collapse digit run into `inf`.
    """

    stripped = text.replace(",", "").strip()
    try:
        return str(int(stripped))
    except ValueError:
        pass
    try:
        value = float(stripped)
    except (ValueError, OverflowError):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return str(int(value)) if value == int(value) else repr(value)


def extract_gold(text: str) -> str | None:
    """The reference solution's final answer, which closes the document."""

    matches = GOLD_PATTERN.findall(text)
    return normalize_number(matches[-1]) if matches else None


def extract_prediction(text: str, fmt: PromptFormat) -> str | None:
    """The model's *first* final answer.

    Taking the last one scores whatever the model drifted into after it had
    already answered, which at this scale is a repetition collapse. The first
    delimiter is where the answer to the asked question is.
    """

    match = answer_pattern(fmt.delimiter).search(text)
    return normalize_number(match.group(1)) if match else None


def truncate_at_stop(text: str, stops) -> str:
    cut = len(text)
    for stop in stops:
        index = text.find(stop)
        if index != -1:
            cut = min(cut, index)
    return text[:cut]


def build_prompt(exemplars, question: str, fmt: PromptFormat, *, strip_calculator: bool) -> str:
    blocks = [
        fmt.exemplar(
            row["question"],
            fmt.solution(row["answer"], strip_calculator=strip_calculator),
        )
        for row in exemplars
    ]
    blocks.append(fmt.query(question))
    return "\n\n".join(blocks)


def select_exemplars(train, shots: int, max_shots: int, seed: int) -> list[int]:
    """Draw one exemplar set per seed, nested across shot counts.

    `random.sample` gives no prefix guarantee, so drawing `max_shots` once and
    slicing is what actually makes the k-shot arms nested rather than an
    accident of CPython's selection-set branch.
    """

    return random.Random(seed).sample(range(len(train)), max_shots)[:shots]


@torch.no_grad()
def greedy_generate(
    model: LatentThoughtModel,
    prompts: list[list[int]],
    *,
    max_new_tokens: int,
    eot_id: int,
    blocked_ids: torch.Tensor,
    device: torch.device,
    stop_check_every: int,
    stops: tuple[str, ...],
    decode,
) -> tuple[list[list[int]], dict[str, int]]:
    """Greedy continuation through one cached prefill and token steps.

    Prompts are left padded for one batched prefill. The key-valid mask keeps
    padded atoms out of dense attention and KDA state. Every later action is a
    one-token cached step; the full prefix is never replayed.
    """

    if stop_check_every < 1:
        raise ValueError("stop_check_every must be at least 1")
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("generation prompts must be nonempty")
    batch = len(prompts)
    prompt_width = max(map(len, prompts))
    width = prompt_width + max_new_tokens
    buffer = torch.full(
        (batch, prompt_width), eot_id, dtype=torch.long, device=device
    )
    key_valid = torch.zeros(
        (batch, prompt_width), dtype=torch.bool, device=device
    )
    for row, ids in enumerate(prompts):
        start = prompt_width - len(ids)
        buffer[row, start:] = torch.tensor(ids, dtype=torch.long, device=device)
        key_valid[row, start:] = True

    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    generated: list[list[int]] = [[] for _ in range(batch)]
    caches = model.make_generation_cache(
        batch, width, device, dtype=torch.bfloat16 if device.type == "cuda" else None
    )
    output = model.prefill(buffer, caches, key_valid)
    logits = output.logits
    decode_steps = 0
    for step in range(max_new_tokens):
        # Ids the corpus never made a target carry unconstrained logits, and
        # argmax breaks ties toward the low id. That covers the padding above
        # the tokenizer's vocabulary and the registered post-training specials,
        # whose projection rows are bit-identical to the padding.
        logits[:, blocked_ids] = float("-inf")
        chosen = logits.argmax(-1).to(torch.long)
        # End-of-text terminates the row and is not part of its answer.
        stopping = chosen == eot_id
        for row, token in enumerate(chosen.tolist()):
            if not finished[row] and not stopping[row]:
                generated[row].append(token)
        finished |= stopping
        if bool(finished.all()):
            break
        if step % stop_check_every == stop_check_every - 1:
            for row in range(batch):
                if not finished[row]:
                    text = decode(generated[row])
                    if any(stop in text for stop in stops):
                        finished[row] = True
            if bool(finished.all()):
                break
        if step + 1 < max_new_tokens:
            # Finished lanes feed EOT into their private caches. Those caches
            # are never read again; live lanes remain exactly independent.
            chosen = torch.where(finished, eot_id, chosen)
            output = model.token_step(chosen, caches, prompt_width + step)
            logits = output.logits
            decode_steps += 1
    return generated, {
        "prefill_forwards": 1,
        "decode_forwards": decode_steps,
        "model_forwards": 1 + decode_steps,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--shots", default="1,5", help="comma-separated exemplar counts to score"
    )
    parser.add_argument(
        "--prompt-format",
        default="harness",
        choices=sorted(PROMPT_FORMATS),
        help="`harness` is the lm-evaluation-harness convention and matches "
        "nothing in the pretraining corpus; `bare` and `dapo` are the two "
        "renderings `openmath_instruct` actually contributed",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="comma-separated exemplar draws; a near-zero headline needs more "
        "than one",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stop-check-every", type=int, default=8)
    parser.add_argument(
        "--keep-calculator-annotations",
        action="store_true",
        help="leave the dataset's <<4*2=8>> spans in the exemplar solutions",
    )
    parser.add_argument(
        "--samples", type=int, default=8, help="generations recorded verbatim"
    )
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    import pandas as pd

    args = build_arg_parser().parse_args()
    device = torch.device("cuda")
    strip_calculator = not args.keep_calculator_annotations
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive; omit it to score everything")
    output = Path(args.output) if args.output else None
    if output is not None and output.exists():
        raise ValueError(
            f"{output} already exists; completed run artifacts are immutable, "
            "so write a new versioned path"
        )

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = dict(payload["model_config"])
    provenance = config.get("tokenizer_provenance")
    if not provenance:
        raise ValueError(
            "checkpoint declares no tokenizer provenance, so its token ids "
            "cannot be tied to a vocabulary"
        )
    tokenizer = load_bound_tokenizer(provenance)
    backbone = load_model(args.checkpoint, device, payload=payload).eval()
    model = LatentThoughtModel(backbone).to(device).eval()

    specials = list(range(len(tokenizer.spec.specials)))
    blocked = sorted(
        set(specials + list(range(tokenizer.vocab_size, config["vocab_size"])))
        - {tokenizer.eot_id}
    )
    blocked_ids = torch.tensor(blocked, dtype=torch.long, device=device)

    train = pd.read_parquet(TRAIN_PARQUET)
    test = pd.read_parquet(TEST_PARQUET)
    if args.limit:
        test = test.iloc[: args.limit]
    gold = [extract_gold(row) for row in test["answer"]]
    if any(value is None for value in gold):
        raise ValueError("a test row has no #### final answer")

    def decode(ids: list[int]) -> str:
        return tokenizer.decode(ids)

    fmt = PROMPT_FORMATS[args.prompt_format]
    shot_counts = [int(k) for k in args.shots.split(",") if k.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    train_seq_len = payload.get("train_seq_len")

    results = []
    for shots in shot_counts:
        for seed in seeds:
            exemplar_rows = select_exemplars(train, shots, max(shot_counts), seed)
            exemplars = [train.iloc[i] for i in exemplar_rows]
            prompts = [
                [tokenizer.eot_id]
                + tokenizer.encode(
                    build_prompt(
                        exemplars, question, fmt,
                        strip_calculator=strip_calculator,
                    ),
                    allow_specials=False,
                )
                for question in test["question"]
            ]
            prompt_lengths = [len(ids) for ids in prompts]
            width = max(prompt_lengths) + args.max_new_tokens
            if train_seq_len and width > train_seq_len:
                raise ValueError(
                    f"{shots}-shot needs {width} positions but the checkpoint "
                    f"trained at {train_seq_len}; the trunk would extrapolate"
                )

            predictions: list[str | None] = []
            answered: list[bool] = []
            transcripts: list[str] = []
            raw_transcripts: list[str] = []
            native_work = {
                "prefill_forwards": 0,
                "decode_forwards": 0,
                "model_forwards": 0,
            }
            generated_token_count = 0
            generation_started = time.perf_counter()
            for start in range(0, len(prompts), args.batch_size):
                chunk = prompts[start : start + args.batch_size]
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    outputs, chunk_work = greedy_generate(
                        model,
                        chunk,
                        max_new_tokens=args.max_new_tokens,
                        eot_id=tokenizer.eot_id,
                        blocked_ids=blocked_ids,
                        device=device,
                        stop_check_every=args.stop_check_every,
                        stops=fmt.stops,
                        decode=decode,
                    )
                for name, value in chunk_work.items():
                    native_work[name] += value
                generated_token_count += sum(map(len, outputs))
                for ids in outputs:
                    raw = decode(ids)
                    text = truncate_at_stop(raw, fmt.stops)
                    raw_transcripts.append(raw)
                    transcripts.append(text)
                    answered.append(fmt.delimiter in text)
                    predictions.append(extract_prediction(text, fmt))
                print(
                    f"shots={shots} seed={seed} "
                    f"scored {len(predictions)}/{len(prompts)}",
                    flush=True,
                )

            generation_seconds = time.perf_counter() - generation_started

            correct = sum(
                1
                for predicted, truth in zip(predictions, gold)
                if predicted is not None and predicted == truth
            )
            record = {
                "shots": shots,
                "seed": seed,
                "prompt_format": fmt.name,
                "exemplar_train_rows": exemplar_rows,
                "examples": len(prompts),
                "exact_match": correct / len(prompts),
                "correct": correct,
                # Separated on purpose: a base model that never writes `####`
                # is failing at format, not arithmetic, and one that writes it
                # but emits an unparseable answer is a third thing again.
                "delimiter_emitted_rate": sum(answered) / len(prompts),
                "parsed_answer_rate": sum(p is not None for p in predictions)
                / len(prompts),
                "mean_prompt_tokens": sum(prompt_lengths) / len(prompt_lengths),
                "max_prompt_tokens": max(prompt_lengths),
                "generation_seconds": generation_seconds,
                "generated_tokens": generated_token_count,
                **native_work,
                "samples": [
                    {
                        "question": q,
                        "gold": g,
                        "prediction": p,
                        "answer": t,
                        "raw_generation": r,
                    }
                    for q, g, p, t, r in list(
                        zip(
                            test["question"],
                            gold,
                            predictions,
                            transcripts,
                            raw_transcripts,
                        )
                    )[: args.samples]
                ],
            }
            print(
                json.dumps({k: v for k, v in record.items() if k != "samples"}),
                flush=True,
            )
            results.append(record)

    by_shots = {
        shots: [r["exact_match"] for r in results if r["shots"] == shots]
        for shots in shot_counts
    }
    summary = {
        "checkpoint": str(args.checkpoint),
        "completed_steps": payload.get("completed_steps"),
        "tokenizer": provenance["name"],
        "prompt_format": fmt.name,
        "gsm8k_train_sha256": sha256(TRAIN_PARQUET),
        "gsm8k_test_sha256": sha256(TEST_PARQUET),
        "seeds": seeds,
        "max_new_tokens": args.max_new_tokens,
        "strip_calculator_annotations": strip_calculator,
        "implementation_maturity": "incremental_kv_kda_cache",
        "mean_exact_match_by_shots": {
            str(shots): sum(values) / len(values) for shots, values in by_shots.items()
        },
        "results": results,
    }
    print(json.dumps(summary["mean_exact_match_by_shots"], sort_keys=True), flush=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
