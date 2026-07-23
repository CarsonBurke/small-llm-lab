"""Build the mixed FineWeb + FineMath-4+ + math-QA pretraining corpus.

Deterministic doc-level interleave into the challenge shard format used by
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
  - OpenMathInstruct-2 (CC-BY-4.0), all published problem sources, one
    problem + generated solution + ``Answer: <x>`` per document.

``--tokenizer gpt2`` builds the same mix under GPT-2 byte-level BPE for the
gpt2vocab nano models: BOS and EOS are both the single ``<|endoftext|>``
token, so QA documents drop their final EOS (the next document's leading
token terminates the last answer).

    python3 build_math_mix_dataset.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import sentencepiece as spm

BOS_ID = 1
EOS_ID = 2
# GPT-2 has a single special token; it serves as both the leading document
# cue (modded-nanogpt shard convention) and the stop signal.
GPT2_EOT_ID = 50256
SHARD_MAGIC = 20240520
SHARD_VERSION = 1
ENCODE_BATCH = 2048
ENCODE_THREADS = 16


class GPT2BatchEncoder:
    """GPT-2 byte-level BPE behind the SentencePiece batch-encode signature.

    Byte-level BPE pre-splits on a regex, so segment boundaries never merge;
    the fast tokenizer parallelizes batches internally, making
    ``num_threads`` advisory only.
    """

    def __init__(self) -> None:
        from transformers import GPT2TokenizerFast

        self._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self._tokenizer.model_max_length = 1 << 30

    def encode(
        self, texts: list[str], out_type: type = int, num_threads: int | None = None
    ) -> list[list[int]]:
        if out_type is not int:
            raise ValueError("GPT2BatchEncoder only encodes to int ids")
        return self._tokenizer(texts)["input_ids"]

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
OPENMATH_SOURCES = {
    "augmented_math",
    "augmented_gsm8k",
    "gsm8k",
    "math",
}

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


def deterministic_fraction(index: int, fraction: float) -> bool:
    """Select exactly ``fraction`` of a deterministic sequence over time."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    exact = Fraction(str(fraction))
    return ((index + 1) * exact.numerator) // exact.denominator > (
        index * exact.numerator
    ) // exact.denominator


def allocate_token_budgets(
    total_tokens: int, fractions: dict[str, float]
) -> dict[str, int]:
    """Largest-remainder allocation whose integer budgets sum exactly."""
    if total_tokens < 0:
        raise ValueError(f"total_tokens must be nonnegative, got {total_tokens}")
    rational_fractions = {
        name: Fraction(str(fraction)) for name, fraction in fractions.items()
    }
    if sum(rational_fractions.values()) != 1:
        raise ValueError("source fractions must sum exactly to 1")
    if any(fraction <= 0 for fraction in rational_fractions.values()):
        raise ValueError("every source fraction must be positive")
    exact = {
        name: total_tokens * fraction
        for name, fraction in rational_fractions.items()
    }
    budgets = {
        name: value.numerator // value.denominator
        for name, value in exact.items()
    }
    remainder = total_tokens - sum(budgets.values())
    order = sorted(
        fractions,
        key=lambda name: (
            exact[name] - budgets[name],
            rational_fractions[name],
        ),
        reverse=True,
    )
    for name in order[:remainder]:
        budgets[name] += 1
    zero_budgets = [name for name, budget in budgets.items() if budget == 0]
    if zero_budgets:
        raise ValueError(
            "total_tokens is too small to allocate every positive source: "
            f"{zero_budgets}"
        )
    return budgets


def least_complete_source(
    names: list[str], budgets: dict[str, int], written: dict[str, int]
) -> str:
    """Deterministically interleave sources by completed budget fraction.

    Unlike sampling documents with token budgets as probabilities, this is
    insensitive to the sources' very different document lengths.  Each
    source iterator is consumed once and no random order is introduced.
    """
    return min(names, key=lambda name: Fraction(written[name], budgets[name]))


def exact_document_stream(
    sources: dict[str, Iterator[np.ndarray]],
    budgets: dict[str, int],
    total_tokens: int,
) -> Iterator[tuple[str, np.ndarray, int]]:
    """Yield each source document once, truncating only the final document."""
    if set(sources) != set(budgets):
        raise ValueError("source and budget names must match")
    if sum(budgets.values()) != total_tokens:
        raise ValueError("source budgets must sum to total_tokens")
    written = {name: 0 for name in sources}
    while sum(written.values()) < total_tokens:
        name = least_complete_source(list(sources), budgets, written)
        document = next(sources[name], None)
        if document is None:
            raise RuntimeError(
                f"source {name!r} exhausted before the exact one-pass corpus "
                f"reached {total_tokens:,} tokens"
            )
        if document.size == 0:
            raise ValueError(f"source {name!r} yielded an empty document")
        remaining = total_tokens - sum(written.values())
        kept_document = document[:remaining]
        truncated_tokens = document.size - kept_document.size
        written[name] += kept_document.size
        yield name, kept_document, truncated_tokens


def write_shard(path: Path, tokens: np.ndarray) -> None:
    header = np.zeros(256, dtype="<i4")
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = tokens.size
    with path.open("wb") as file:
        file.write(header.tobytes())
        file.write(tokens.astype("<u2", copy=False).tobytes())


def fineweb_documents(
    dataset_dir: Path, bos_id: int = BOS_ID
) -> Iterator[np.ndarray]:
    """Already-tokenized docs, recovered by splitting the stream at BOS.

    The source shards pack documents across file boundaries, so the segment
    after a file's last BOS is carried into the next file instead of being
    yielded as a truncated document.
    """
    carry = np.empty(0, dtype=np.int32)
    first_shard = True
    for shard in sorted(dataset_dir.glob("fineweb_train_*.bin")):
        tokens = np.fromfile(shard, dtype="<u2", offset=256 * 4).astype(np.int32)
        stream = np.concatenate((carry, tokens)) if carry.size else tokens
        starts = np.flatnonzero(stream == bos_id)
        if first_shard and starts.size == 0:
            # A full shard without a single document boundary means the
            # shards were tokenized under a different vocabulary than the
            # requested --tokenizer implies.
            raise ValueError(
                f"{shard} contains no BOS id {bos_id}; --fineweb-dataset "
                "and --tokenizer disagree on the vocabulary"
            )
        first_shard = False
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
    texts: Iterator[str],
    tokenizer: spm.SentencePieceProcessor | GPT2BatchEncoder,
    bos_id: int = BOS_ID,
) -> Iterator[np.ndarray]:
    """Batch-encode a text stream into BOS-prefixed token documents."""
    batch: list[str] = []

    def encode(chunk: list[str]) -> Iterator[np.ndarray]:
        encoded = tokenizer.encode(chunk, out_type=int, num_threads=ENCODE_THREADS)
        for pieces in encoded:
            yield np.asarray([bos_id] + pieces, dtype=np.int32)

    for text in texts:
        batch.append(text)
        if len(batch) == ENCODE_BATCH:
            yield from encode(batch)
            batch = []
    if batch:
        yield from encode(batch)


def encoded_qa_documents(
    segmented: Iterator[list[str]],
    tokenizer: spm.SentencePieceProcessor | GPT2BatchEncoder,
    bos_id: int = BOS_ID,
    eos_id: int = EOS_ID,
) -> Iterator[np.ndarray]:
    """Encode QA documents with EOS closing every answer segment.

    The SentencePiece model normalizes newlines away, so EOS is the only
    stop signal the model can learn; RL rollouts and the AIME eval truncate
    generations at the first EOS, which makes the verifier's ``Answer: <x>``
    extraction terminal instead of budget-bounded.

    When BOS and EOS are the same token (GPT-2's single ``<|endoftext|>``),
    the document's final EOS is dropped: the next document's leading token
    already terminates the last answer, and keeping both would pretrain a
    doubled stop token the RL rollouts never emit.
    """
    batch: list[list[str]] = []

    def encode(chunk: list[list[str]]) -> Iterator[np.ndarray]:
        flat = [segment for document in chunk for segment in document]
        encoded = tokenizer.encode(flat, out_type=int, num_threads=ENCODE_THREADS)
        cursor = 0
        for document in chunk:
            tokens: list[int] = [bos_id]
            for _ in document:
                tokens.extend(encoded[cursor])
                tokens.append(eos_id)
                cursor += 1
            if bos_id == eos_id:
                tokens.pop()
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


def deepmind_qa_pairs(easy_dir: Path) -> Iterator[tuple[str, str]]:
    """QA pairs round-robined across module files without random ordering."""
    handles = [
        (easy_dir / f"{module}.txt").open(encoding="utf-8")
        for module in DEEPMIND_EASY_MODULES
    ]
    try:
        cursor = 0
        while handles:
            handle = handles[cursor]
            question = handle.readline()
            answer = handle.readline()
            if not answer:
                if question.strip():
                    print(f"warning: dropping unpaired trailing question {question!r}")
                handles.pop(cursor)
                handle.close()
                if handles:
                    cursor %= len(handles)
                continue
            yield question.strip(), answer.strip()
            cursor = (cursor + 1) % len(handles)
    finally:
        for handle in handles:
            handle.close()


def deepmind_worksheets(
    easy_dir: Path,
    template_fraction: float = 0.0,
    min_problems: int = 8,
    max_problems: int = 24,
) -> Iterator[list[str]]:
    """Worksheet documents of short QA segments, one EOS-closed per problem.

    A ``template_fraction`` share of documents is instead a single problem
    wrapped in the verbatim DAPO template, matching the RL prompt shape.
    """
    pairs = deepmind_qa_pairs(easy_dir)
    document_index = 0
    while True:
        templated = deterministic_fraction(document_index, template_fraction)
        problem_count = min_problems + document_index % (max_problems - min_problems + 1)
        document_index += 1
        if templated:
            pair = next(pairs, None)
            if pair is None:
                return
            question, answer = pair
            yield [dapo_wrapped(question, f"Answer: {answer}")]
            continue
        worksheet = []
        for _ in range(problem_count):
            pair = next(pairs, None)
            if pair is None:
                break
            question, answer = pair
            worksheet.append(f"{question}\nAnswer: {answer}")
        if not worksheet:
            return
        yield worksheet


def openmath_documents(
    parquet_files: list[Path], template_fraction: float = 0.0
) -> Iterator[list[str]]:
    """One EOS-closed problem + generated solution + Answer per document.

    A ``template_fraction`` share of documents wraps the problem in the
    verbatim DAPO template, teaching step-by-step then ``Answer:`` under the
    exact instruction the RL prompts use.
    """
    columns = ["problem", "generated_solution", "expected_answer", "problem_source"]
    document_index = 0
    for parquet_path in sorted(parquet_files):
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=ENCODE_BATCH, columns=columns):
            for row in batch.to_pylist():
                if row["problem_source"] not in OPENMATH_SOURCES:
                    continue
                fields = (row["problem"], row["generated_solution"], row["expected_answer"])
                if any(field is None for field in fields):
                    print(f"warning: skipping row with null field in {parquet_path.name}")
                    continue
                problem, solution, answer = (field.strip() for field in fields)
                response = f"{solution}\nAnswer: {answer}"
                if deterministic_fraction(document_index, template_fraction):
                    yield [dapo_wrapped(problem, response)]
                else:
                    yield [f"{problem}\n{response}"]
                document_index += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/datasets/mathmix_v4_sp1024")
    parser.add_argument("--fineweb-dataset", default="data/datasets/fineweb10B_sp1024")
    parser.add_argument("--finemath-dir", default="postraining/data/finemath")
    parser.add_argument(
        "--deepmind-easy-dir",
        default="postraining/data/mathematics_dataset-v1.0/train-easy",
    )
    parser.add_argument("--openmath-dir", default="postraining/data/openmathinstruct2")
    parser.add_argument(
        "--tokenizer",
        default="data/tokenizers/fineweb_1024_bpe.model",
        help="SentencePiece model path, or the sentinel 'gpt2' for GPT-2 BPE "
        "(BOS and EOS both become <|endoftext|> = 50256)",
    )
    parser.add_argument("--total-tokens", type=int, default=None)
    parser.add_argument("--training-steps", type=int, default=2000)
    parser.add_argument("--train-batch-tokens", type=int, default=524_288)
    # train_gpt fixes world_size * grad_accum_steps at 8. Each loader span
    # consumes one extra next-token target, hence +8 stream tokens per step.
    parser.add_argument("--loader-spans-per-step", type=int, default=8)
    parser.add_argument("--fineweb-fraction", type=float, default=0.45)
    parser.add_argument("--finemath-fraction", type=float, default=0.25)
    parser.add_argument("--deepmind-fraction", type=float, default=0.21)
    parser.add_argument("--openmath-fraction", type=float, default=0.09)
    parser.add_argument("--shard-size", type=int, default=100_000_000)
    parser.add_argument("--template-fraction", type=float, default=0.5)
    args = parser.parse_args()

    fractions = {
        "fineweb": args.fineweb_fraction,
        "finemath": args.finemath_fraction,
        "deepmind_easy": args.deepmind_fraction,
        "openmath": args.openmath_fraction,
    }
    positive_counts = {
        "training_steps": args.training_steps,
        "train_batch_tokens": args.train_batch_tokens,
        "loader_spans_per_step": args.loader_spans_per_step,
        "shard_size": args.shard_size,
    }
    invalid_counts = {
        name: value for name, value in positive_counts.items() if value <= 0
    }
    if invalid_counts:
        parser.error(f"counts must be positive: {invalid_counts}")
    rational_fraction_sum = sum(Fraction(str(value)) for value in fractions.values())
    if rational_fraction_sum != 1:
        parser.error("source fractions must sum exactly to 1")
    if any(fraction < 0.0 for fraction in fractions.values()):
        parser.error("source fractions must not be negative")
    fractions = {name: value for name, value in fractions.items() if value > 0.0}
    if not 0.0 <= args.template_fraction <= 1.0:
        parser.error("--template-fraction must be in [0, 1]")
    required_stream_tokens = args.training_steps * (
        args.train_batch_tokens + args.loader_spans_per_step
    )
    total_tokens = required_stream_tokens
    if args.total_tokens is not None and args.total_tokens != required_stream_tokens:
        parser.error(
            "an exact one-pass corpus must match the loader geometry: "
            f"--total-tokens must be {required_stream_tokens:,}, got "
            f"{args.total_tokens:,}"
        )

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
    if "deepmind_easy" in fractions:
        missing = [
            module
            for module in DEEPMIND_EASY_MODULES
            if not (easy_dir / f"{module}.txt").exists()
        ]
        if missing:
            parser.error(f"missing DeepMind module files: {missing}")
    needs_encoder = bool(fractions.keys() - {"fineweb"})
    if args.tokenizer == "gpt2":
        bos_id = eos_id = GPT2_EOT_ID
        tokenizer = GPT2BatchEncoder() if needs_encoder else None
    else:
        bos_id, eos_id = BOS_ID, EOS_ID
        tokenizer = (
            spm.SentencePieceProcessor(model_file=args.tokenizer)
            if needs_encoder
            else None
        )
    source_builders: dict[str, Callable[[], Iterator[np.ndarray]]] = {
        "fineweb": lambda: fineweb_documents(Path(args.fineweb_dataset), bos_id),
        "finemath": lambda: encoded_documents(
            finemath_texts(sorted(Path(args.finemath_dir).rglob("*.parquet"))),
            tokenizer,
            bos_id,
        ),
        "deepmind_easy": lambda: encoded_qa_documents(
            deepmind_worksheets(easy_dir, args.template_fraction),
            tokenizer,
            bos_id,
            eos_id,
        ),
        "openmath": lambda: encoded_qa_documents(
            openmath_documents(
                sorted(Path(args.openmath_dir).rglob("*.parquet")),
                args.template_fraction,
            ),
            tokenizer,
            bos_id,
            eos_id,
        ),
    }
    sources: dict[str, Iterator[np.ndarray]] = {
        name: source_builders[name]() for name in fractions
    }
    budgets = allocate_token_budgets(total_tokens, fractions)
    written_tokens = {name: 0 for name in sources}
    written_docs = {name: 0 for name in sources}

    buffer = np.empty(args.shard_size, dtype=np.uint16)
    fill = 0
    shard_index = 0
    shard_source_tokens = {name: 0 for name in sources}
    shard_stats: list[dict[str, object]] = []

    def flush() -> None:
        nonlocal fill, shard_index, shard_source_tokens
        if fill:
            if sum(shard_source_tokens.values()) != fill:
                raise AssertionError("per-source shard counts do not match payload")
            write_shard(
                output_dir / f"fineweb_train_{shard_index:06d}.bin", buffer[:fill]
            )
            shard_stats.append(
                {
                    "index": shard_index,
                    "tokens": fill,
                    "source_tokens": dict(shard_source_tokens),
                }
            )
            shard_index += 1
            fill = 0
            shard_source_tokens = {name: 0 for name in sources}

    final_document_truncated_tokens = 0
    for name, kept_document, truncated_tokens in exact_document_stream(
        sources, budgets, total_tokens
    ):
        final_document_truncated_tokens = truncated_tokens
        written_tokens[name] += kept_document.size
        written_docs[name] += 1
        position = 0
        while position < kept_document.size:
            take = min(args.shard_size - fill, kept_document.size - position)
            buffer[fill : fill + take] = kept_document[position : position + take]
            fill += take
            shard_source_tokens[name] += take
            position += take
            if fill == args.shard_size:
                flush()
    flush()

    if sum(written_tokens.values()) != total_tokens:
        raise AssertionError("global source counts do not match exact corpus size")
    shard_totals = {name: 0 for name in sources}
    for shard in shard_stats:
        for name, count in shard["source_tokens"].items():
            shard_totals[name] += count
    if shard_totals != written_tokens:
        raise AssertionError("per-shard source counts do not match global counts")

    for val_shard in sorted(Path(args.fineweb_dataset).glob("fineweb_val_*.bin")):
        shutil.copy2(val_shard, output_dir / val_shard.name)

    manifest = {
        "total_tokens": sum(written_tokens.values()),
        "requested_total_tokens": total_tokens,
        "required_stream_tokens": required_stream_tokens,
        "training_steps": args.training_steps,
        "train_batch_tokens": args.train_batch_tokens,
        "loader_spans_per_step": args.loader_spans_per_step,
        "one_pass": True,
        "train_shards": shard_index,
        "shard_size": args.shard_size,
        "fractions": fractions,
        "tokens": written_tokens,
        "documents": written_docs,
        "achieved_fractions": {
            name: count / total_tokens for name, count in written_tokens.items()
        },
        "shards": shard_stats,
        "final_document_truncated_tokens": final_document_truncated_tokens,
        "deepmind_modules": DEEPMIND_EASY_MODULES,
        "openmath_sources": sorted(OPENMATH_SOURCES),
        "bos_id": bos_id,
        "eos_id": eos_id,
        "qa_eos": (
            "EOS appended after every Answer segment in QA sources"
            + (
                "; document-final EOS dropped (next document's BOS terminates it)"
                if bos_id == eos_id
                else ""
            )
        ),
        "template_fraction": args.template_fraction,
        "template": "verbatim DAPO-Math-17K prompt wrapper on a share of QA docs",
        "tokenizer": args.tokenizer,
        "ordering": "deterministic_least_completed_token_budget_no_rng",
        "rng": "none",
        "val": "copied unchanged from FineWeb (BPB comparability)",
    }
    (output_dir / "mix_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
