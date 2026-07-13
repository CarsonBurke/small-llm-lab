"""Build the mixed FineWeb + FineMath-4+ + math-QA pretraining corpus.

Doc-level weighted interleave into the challenge shard format used by
``train_gpt.py`` (256-int32 header ``[20240520, 1, num_tokens]`` followed by
uint16 tokens; each document is ``[BOS] + pieces`` with no EOS), matching
``data/download_hf_docs_and_tokenize.py``.  The validation shard is copied
unchanged from the FineWeb dataset so BPB stays comparable across ablations.

Sources (all open, none homemade):
- FineWeb docs, recovered from the existing tokenized shards by BOS split.
- FineMath-4+ parquet (HuggingFaceTB/finemath), math web prose.
- QA slice, every answer closed with ``Answer: <x>`` followed by EOS (the
  tokenizer normalizes newlines away, so EOS is the model's only learnable
  stop signal; FineWeb/FineMath docs keep the corpus convention of no EOS):
  - DeepMind mathematics_dataset v1.0 train-easy (Apache-2.0), worksheet
    documents of many short QA pairs from verifier-compatible modules;
  - OpenMathInstruct-2 (CC-BY-4.0) gsm8k/augmented_gsm8k word problems,
    one problem + short solution + ``Answer: <x>`` per document.

    python3 build_math_mix_dataset.py --total-tokens 500_000_000
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import sentencepiece as spm

BOS_ID = 1
EOS_ID = 2
SHARD_MAGIC = 20240520
SHARD_VERSION = 1
ENCODE_BATCH = 2048
ENCODE_THREADS = 16

# train-easy modules whose answers survive the Minerva-style verifier
# normalization verbatim (integers, small fractions/decimals, sorted lists,
# choice letters, True/False) — no unit-bearing measurements, no surds.
DEEPMIND_EASY_MODULES = [
    "arithmetic__add_or_sub",
    "arithmetic__add_sub_multiple",
    "arithmetic__mul",
    "arithmetic__div",
    "arithmetic__mul_div_multiple",
    "arithmetic__mixed",
    "comparison__closest",
    "comparison__kth_biggest",
    "comparison__pair",
    "comparison__sort",
    "numbers__place_value",
    "numbers__div_remainder",
    "numbers__gcd",
    "numbers__lcm",
    "numbers__round_number",
    "algebra__linear_1d",
    "algebra__linear_2d",
    "algebra__sequence_next_term",
]
OPENMATH_EASY_SOURCES = {"gsm8k", "augmented_gsm8k"}

# The verbatim DAPO-Math-17K prompt wrapper.  RL prompts arrive in exactly
# this template, and v2 showed the model treats template-wrapped problems as
# prose to continue (0/1024 rollouts produced an ``Answer:`` line), so a
# fraction of QA documents pretrain the mapping template -> answer -> EOS.
DAPO_PREAMBLE = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: $Answer (without quotes) where "
    "$Answer is the answer to the problem."
)
DAPO_REMINDER = 'Remember to put your answer on its own line after "Answer:".'


def dapo_wrapped(problem: str, response: str) -> str:
    """One DAPO-templated document: the RL prompt followed by its response."""
    return f"{DAPO_PREAMBLE}\n\n{problem}\n\n{DAPO_REMINDER}\n{response}"


def write_shard(path: Path, tokens: np.ndarray) -> None:
    header = np.zeros(256, dtype="<i4")
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = tokens.size
    with path.open("wb") as file:
        file.write(header.tobytes())
        file.write(tokens.astype("<u2", copy=False).tobytes())


def fineweb_documents(dataset_dir: Path) -> Iterator[np.ndarray]:
    """Already-tokenized docs, recovered by splitting the stream at BOS.

    The source shards pack documents across file boundaries, so the segment
    after a file's last BOS is carried into the next file instead of being
    yielded as a truncated document.
    """
    carry = np.empty(0, dtype=np.int32)
    for shard in sorted(dataset_dir.glob("fineweb_train_*.bin")):
        tokens = np.fromfile(shard, dtype="<u2", offset=256 * 4).astype(np.int32)
        stream = np.concatenate((carry, tokens)) if carry.size else tokens
        starts = np.flatnonzero(stream == BOS_ID)
        if starts.size == 0:
            carry = stream
            continue
        # Only the very first file can start mid-document; afterwards the
        # carry always begins at a BOS.
        stream = stream[starts[0]:]
        starts = starts - starts[0]
        for begin, end in zip(starts[:-1], starts[1:]):
            yield stream[begin:end]
        carry = stream[starts[-1]:]
    if carry.size:
        yield carry


def encoded_documents(
    texts: Iterator[str], tokenizer: spm.SentencePieceProcessor
) -> Iterator[np.ndarray]:
    """Batch-encode a text stream into BOS-prefixed token documents."""
    batch: list[str] = []

    def encode(chunk: list[str]) -> Iterator[np.ndarray]:
        encoded = tokenizer.encode(chunk, out_type=int, num_threads=ENCODE_THREADS)
        for pieces in encoded:
            yield np.asarray([BOS_ID] + pieces, dtype=np.int32)

    for text in texts:
        batch.append(text)
        if len(batch) == ENCODE_BATCH:
            yield from encode(batch)
            batch = []
    if batch:
        yield from encode(batch)


def encoded_qa_documents(
    segmented: Iterator[list[str]], tokenizer: spm.SentencePieceProcessor
) -> Iterator[np.ndarray]:
    """Encode QA documents with EOS closing every answer segment.

    The SentencePiece model normalizes newlines away, so EOS is the only
    stop signal the model can learn; RL rollouts and the AIME eval truncate
    generations at the first EOS, which makes the verifier's ``Answer: <x>``
    extraction terminal instead of budget-bounded.
    """
    batch: list[list[str]] = []

    def encode(chunk: list[list[str]]) -> Iterator[np.ndarray]:
        flat = [segment for document in chunk for segment in document]
        encoded = tokenizer.encode(flat, out_type=int, num_threads=ENCODE_THREADS)
        cursor = 0
        for document in chunk:
            tokens: list[int] = [BOS_ID]
            for _ in document:
                tokens.extend(encoded[cursor])
                tokens.append(EOS_ID)
                cursor += 1
            yield np.asarray(tokens, dtype=np.int32)

    for segments in segmented:
        batch.append(segments)
        if len(batch) == ENCODE_BATCH:
            yield from encode(batch)
            batch = []
    if batch:
        yield from encode(batch)


def finemath_texts(parquet_files: list[Path]) -> Iterator[str]:
    for parquet_path in sorted(parquet_files):
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=ENCODE_BATCH, columns=["text"]):
            yield from batch["text"].to_pylist()


def deepmind_qa_pairs(easy_dir: Path, rng: random.Random) -> Iterator[tuple[str, str]]:
    """QA pairs interleaved across module files (each file: Q line, A line)."""
    handles = [
        (easy_dir / f"{module}.txt").open(encoding="utf-8")
        for module in DEEPMIND_EASY_MODULES
    ]
    try:
        while handles:
            handle = rng.choice(handles)
            question = handle.readline()
            answer = handle.readline()
            if not answer:
                if question.strip():
                    print(f"warning: dropping unpaired trailing question {question!r}")
                handles.remove(handle)
                handle.close()
                continue
            yield question.strip(), answer.strip()
    finally:
        for handle in handles:
            handle.close()


def deepmind_worksheets(
    easy_dir: Path,
    rng: random.Random,
    template_fraction: float = 0.0,
    min_problems: int = 8,
    max_problems: int = 24,
) -> Iterator[list[str]]:
    """Worksheet documents of short QA segments, one EOS-closed per problem.

    A ``template_fraction`` share of documents is instead a single problem
    wrapped in the verbatim DAPO template, matching the RL prompt shape.
    """
    pairs = deepmind_qa_pairs(easy_dir, rng)
    while True:
        if rng.random() < template_fraction:
            pair = next(pairs, None)
            if pair is None:
                return
            question, answer = pair
            yield [dapo_wrapped(question, f"Answer: {answer}")]
            continue
        worksheet = []
        for _ in range(rng.randint(min_problems, max_problems)):
            pair = next(pairs, None)
            if pair is None:
                break
            question, answer = pair
            worksheet.append(f"{question}\nAnswer: {answer}")
        if not worksheet:
            return
        yield worksheet


def openmath_documents(
    parquet_files: list[Path], rng: random.Random, template_fraction: float = 0.0
) -> Iterator[list[str]]:
    """One EOS-closed problem + short solution + Answer line per document.

    A ``template_fraction`` share of documents wraps the problem in the
    verbatim DAPO template, teaching step-by-step then ``Answer:`` under the
    exact instruction the RL prompts use.
    """
    columns = ["problem", "generated_solution", "expected_answer", "problem_source"]
    for parquet_path in sorted(parquet_files):
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=ENCODE_BATCH, columns=columns):
            for row in batch.to_pylist():
                if row["problem_source"] not in OPENMATH_EASY_SOURCES:
                    continue
                fields = (row["problem"], row["generated_solution"], row["expected_answer"])
                if any(field is None for field in fields):
                    print(f"warning: skipping row with null field in {parquet_path.name}")
                    continue
                problem, solution, answer = (field.strip() for field in fields)
                response = f"{solution}\nAnswer: {answer}"
                if rng.random() < template_fraction:
                    yield [dapo_wrapped(problem, response)]
                else:
                    yield [f"{problem}\n{response}"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/datasets/mathmix_v3_sp1024")
    parser.add_argument("--fineweb-dataset", default="data/datasets/fineweb10B_sp1024")
    parser.add_argument("--finemath-dir", default="postraining/data/finemath")
    parser.add_argument(
        "--deepmind-easy-dir",
        default="postraining/data/mathematics_dataset-v1.0/train-easy",
    )
    parser.add_argument("--openmath-dir", default="postraining/data/openmathinstruct2")
    parser.add_argument("--tokenizer", default="data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--total-tokens", type=int, default=500_000_000)
    parser.add_argument("--fineweb-fraction", type=float, default=0.45)
    parser.add_argument("--finemath-fraction", type=float, default=0.25)
    parser.add_argument("--deepmind-fraction", type=float, default=0.21)
    parser.add_argument("--openmath-fraction", type=float, default=0.09)
    parser.add_argument("--shard-size", type=int, default=100_000_000)
    parser.add_argument("--template-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    fractions = {
        "fineweb": args.fineweb_fraction,
        "finemath": args.finemath_fraction,
        "deepmind_easy": args.deepmind_fraction,
        "openmath_gsm": args.openmath_fraction,
    }
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        parser.error("source fractions must sum to 1")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    stale = sorted(output_dir.glob("fineweb_*.bin"))
    if stale:
        # The training loader globs every matching file, so leftovers from a
        # previous build would silently contaminate the corpus.  Deleting user
        # data is not this script's call — fail fast instead.
        parser.error(
            f"{output_dir} already holds {len(stale)} shard(s) "
            f"(e.g. {stale[0].name}); remove them or pick a fresh --output"
        )
    easy_dir = Path(args.deepmind_easy_dir)
    missing = [
        module
        for module in DEEPMIND_EASY_MODULES
        if not (easy_dir / f"{module}.txt").exists()
    ]
    if missing:
        parser.error(f"missing DeepMind module files: {missing}")
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    rng = random.Random(args.seed)

    sources: dict[str, Iterator[np.ndarray]] = {
        "fineweb": fineweb_documents(Path(args.fineweb_dataset)),
        "finemath": encoded_documents(
            finemath_texts(sorted(Path(args.finemath_dir).rglob("*.parquet"))),
            tokenizer,
        ),
        "deepmind_easy": encoded_qa_documents(
            deepmind_worksheets(easy_dir, rng, args.template_fraction), tokenizer
        ),
        "openmath_gsm": encoded_qa_documents(
            openmath_documents(
                sorted(Path(args.openmath_dir).rglob("*.parquet")),
                rng,
                args.template_fraction,
            ),
            tokenizer,
        ),
    }
    remaining = {
        name: int(round(fraction * args.total_tokens))
        for name, fraction in fractions.items()
    }
    written_tokens = {name: 0 for name in sources}
    written_docs = {name: 0 for name in sources}

    buffer = np.empty(args.shard_size, dtype=np.uint16)
    fill = 0
    shard_index = 0

    def flush() -> None:
        nonlocal fill, shard_index
        if fill:
            write_shard(
                output_dir / f"fineweb_train_{shard_index:06d}.bin", buffer[:fill]
            )
            shard_index += 1
            fill = 0

    unmet_tokens: dict[str, int] = {}
    # Documents are packed atomically, so each source can overshoot its
    # budget by at most one document; --total-tokens is a floor, not exact.
    while any(budget > 0 for budget in remaining.values()):
        names = [name for name, budget in remaining.items() if budget > 0]
        name = rng.choices(names, weights=[remaining[n] for n in names], k=1)[0]
        document = next(sources[name], None)
        if document is None:
            print(f"{name}: exhausted with {remaining[name]:,} tokens unmet")
            unmet_tokens[name] = remaining[name]
            remaining[name] = 0
            continue
        remaining[name] -= document.size
        written_tokens[name] += document.size
        written_docs[name] += 1
        position = 0
        while position < document.size:
            take = min(args.shard_size - fill, document.size - position)
            buffer[fill : fill + take] = document[position : position + take]
            fill += take
            position += take
            if fill == args.shard_size:
                flush()
    flush()

    for val_shard in sorted(Path(args.fineweb_dataset).glob("fineweb_val_*.bin")):
        shutil.copy2(val_shard, output_dir / val_shard.name)

    manifest = {
        "total_tokens": sum(written_tokens.values()),
        "train_shards": shard_index,
        "shard_size": args.shard_size,
        "fractions": fractions,
        "tokens": written_tokens,
        "documents": written_docs,
        "unmet_tokens": unmet_tokens,
        "deepmind_modules": DEEPMIND_EASY_MODULES,
        "openmath_sources": sorted(OPENMATH_EASY_SOURCES),
        "qa_eos": "EOS appended after every Answer segment in QA sources",
        "template_fraction": args.template_fraction,
        "template": "verbatim DAPO-Math-17K prompt wrapper on a share of QA docs",
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "val": "copied unchanged from FineWeb (BPB comparability)",
    }
    (output_dir / "mix_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
