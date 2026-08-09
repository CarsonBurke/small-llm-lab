"""Train a combined ToaST + TST tokenizer from a corpus sample.

Pipeline, following the paper's sections 2, 4 and 5.1:

1. route numeric spans to TST and remove them from the text stream, so no
   digit fragment can ever win vocabulary budget;
2. pre-tokenize the remainder and aggregate unique pretokens by count;
3. count byte n-grams within pretoken boundaries above ``--min-count``;
4. build one split tree per retained pretoken;
5. solve the vocabulary integer program for the text budget that remains after
   the special and numeric tokens are reserved;
6. write the spec and the count dictionary the encoder needs at inference.

This is a CPU workload, but it still goes through ``mlq`` under the repository's
single-workload queue policy.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import regex

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenization import ngram_store
from tokenization.spec import (
    DEFAULT_SPECIALS,
    PRETOKEN_PATTERN,
    NgramReference,
    TokenizerSpec,
)
from tokenization.split_tree import build_split_tree, count_ngrams
from tokenization.tokenizer import SplitTreeNumericTokenizer
from tokenization.tst import NUMERIC_SPAN_RE, NumericScheme
from tokenization.vocab_lp import select_vocabulary

BYTE_ALPHABET = tuple(bytes([value]) for value in range(256))


def read_documents(paths: list[Path], limit_bytes: int | None) -> Iterator[str]:
    """Documents from ``.jsonl`` (a ``text`` field) or ``.txt`` (one per line)."""
    seen = 0
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if path.name.endswith((".jsonl", ".jsonl.gz")):
                    text = json.loads(line).get("text", "")
                else:
                    text = line
                if not text:
                    continue
                seen += len(text.encode("utf-8"))
                yield text
                if limit_bytes is not None and seen >= limit_bytes:
                    return


def pretoken_counts(
    documents: Iterator[str], scheme: NumericScheme, pattern: str
) -> tuple[Counter, int, int]:
    """Count pretokens after numeric spans are removed.

    Returns the counts, the total corpus bytes seen, and the number of bytes
    that TST claimed, which is the share of the corpus the text vocabulary no
    longer has to model.
    """
    compiled = regex.compile(pattern)
    counts: Counter = Counter()
    total_bytes = 0
    numeric_bytes = 0
    for text in documents:
        total_bytes += len(text.encode("utf-8"))
        position = 0
        segments: list[str] = []
        for match in NUMERIC_SPAN_RE.finditer(text):
            integer, fraction = scheme.split_span(match.group())
            if not scheme.accepts(integer, fraction):
                continue
            if match.start() > position:
                segments.append(text[position : match.start()])
            numeric_bytes += len(match.group())
            position = match.end()
        if position < len(text):
            segments.append(text[position:])
        for segment in segments:
            for match in compiled.finditer(segment):
                counts[match.group().encode("utf-8")] += 1
    return counts, total_bytes, numeric_bytes


def build_tokenizer(
    *,
    documents: Iterator[str],
    scheme: NumericScheme,
    vocab_size: int,
    specials: tuple[str, ...],
    min_count: int,
    max_ngram: int,
    max_trees: int,
    pattern: str,
    log,
) -> tuple[TokenizerSpec, dict[bytes, int], dict]:
    started = time.monotonic()
    counts, total_bytes, numeric_bytes = pretoken_counts(documents, scheme, pattern)
    log(
        f"pretokens: {len(counts):,} unique, {sum(counts.values()):,} total; "
        f"corpus {total_bytes / 1e6:.1f} MB, TST claimed "
        f"{numeric_bytes / max(total_bytes, 1):.2%} of bytes"
    )
    if not counts:
        raise ValueError("corpus sample produced no pretokens")

    ranked = counts.most_common()
    retained = ranked[:max_trees]
    covered = sum(count for _, count in retained)
    total = sum(counts.values())
    log(
        f"split trees: {len(retained):,} pretokens covering "
        f"{covered / total:.2%} of pretoken occurrences"
    )

    grams = count_ngrams(retained, min_count=min_count, max_length=max_ngram)
    # `most_known_split` walks prefixes, and the candidate set must contain the
    # byte alphabet, so every single byte is present regardless of frequency.
    for byte in BYTE_ALPHABET:
        grams.setdefault(byte, min_count)
    log(f"n-grams: {len(grams):,} at min_count={min_count}, max_length={max_ngram}")

    trees = []
    tree_counts = []
    for pretoken, count in retained:
        trees.append(build_split_tree(pretoken, grams))
        tree_counts.append(count)
    nodes = sum(len(tree.nodes) for tree in trees)
    log(f"tree nodes: {nodes:,}")

    probe = TokenizerSpec(
        numeric=scheme,
        text_tokens=(),
        ngrams=NgramReference(
            filename="ngrams.bin", sha256="", entries=0, min_count=0, max_length=0
        ),
        specials=specials,
        pretoken_pattern=pattern,
    )
    text_budget = vocab_size - probe.text_base
    if text_budget < len(BYTE_ALPHABET):
        raise ValueError(
            f"vocab_size {vocab_size} leaves {text_budget} text tokens, below "
            f"the {len(BYTE_ALPHABET)}-byte alphabet; the numeric scheme "
            f"reserves {probe.numeric_count} tokens"
        )
    log(
        f"budget: {vocab_size} total = {len(specials)} special + "
        f"{probe.numeric_count} numeric + {text_budget} text"
    )

    solution = select_vocabulary(
        trees,
        tree_counts,
        size=text_budget,
        forced=BYTE_ALPHABET,
    )
    log(
        f"LP: objective {solution.lp_objective:,.0f}, rounded "
        f"{solution.rounded_objective:,.0f}, gap {solution.relative_gap:.2e}, "
        f"{solution.fractional_variables} fractional x"
    )

    spec = TokenizerSpec(
        numeric=scheme,
        text_tokens=solution.vocabulary,
        ngrams=NgramReference(
            filename="ngrams.bin",
            sha256="",
            entries=len(grams),
            min_count=min_count,
            max_length=max_ngram,
        ),
        specials=specials,
        pretoken_pattern=pattern,
        provenance={
            "corpus_bytes": total_bytes,
            "numeric_bytes": numeric_bytes,
            "unique_pretokens": len(counts),
            "split_trees": len(retained),
            "pretoken_coverage": covered / total,
            "tree_nodes": nodes,
            "lp_objective": solution.lp_objective,
            "rounded_objective": solution.rounded_objective,
            "lp_relative_gap": solution.relative_gap,
            "fractional_variables": solution.fractional_variables,
            "train_seconds": time.monotonic() - started,
        },
    )
    stats = {
        "tokens_on_training_pretokens": solution.rounded_objective,
        "pretoken_occurrences": total,
    }
    return spec, grams, stats


def evaluate(
    tokenizer: SplitTreeNumericTokenizer, documents: list[str]
) -> dict[str, float]:
    """Bytes per token and exact round-trip on held-out text."""
    total_bytes = 0
    total_tokens = 0
    failures = 0
    for text in documents:
        ids = tokenizer.encode(text)
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(ids)
        if tokenizer.decode(ids) != text:
            failures += 1
    return {
        "bytes": total_bytes,
        "tokens": total_tokens,
        "bytes_per_token": total_bytes / max(total_tokens, 1),
        "roundtrip_failures": failures,
        "documents": len(documents),
    }


def gpt2_baseline(documents: list[str]) -> dict[str, float]:
    """The same measurement under GPT-2, for a directly comparable number.

    A tokenizer's bytes per token means nothing on its own; it means something
    against the tokenizer it would replace, measured on the same text. Without
    this the ablation's compression claim would rest on two numbers taken from
    two different corpora.
    """
    from transformers import GPT2TokenizerFast

    gpt2 = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt2.model_max_length = 1 << 30
    total_bytes = sum(len(text.encode("utf-8")) for text in documents)
    total_tokens = sum(len(ids) for ids in gpt2(documents)["input_ids"])
    return {
        "bytes": total_bytes,
        "tokens": total_tokens,
        "bytes_per_token": total_bytes / max(total_tokens, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument(
        "--option-a",
        action="store_true",
        help="separate digit-group and magnitude tokens instead of compound",
    )
    parser.add_argument("--max-int-digits", type=int, default=19)
    parser.add_argument("--max-frac-digits", type=int, default=15)
    parser.add_argument("--min-count", type=int, default=250)
    parser.add_argument("--max-ngram", type=int, default=32)
    parser.add_argument("--max-trees", type=int, default=60_000)
    parser.add_argument("--limit-bytes", type=int, default=None)
    parser.add_argument("--validation-limit-bytes", type=int, default=4_000_000)
    parser.add_argument(
        "--baseline-gpt2",
        action="store_true",
        help="also measure GPT-2 on the validation text, so the compression "
        "claim is a comparison rather than a bare number",
    )
    args = parser.parse_args()

    def log(message: str) -> None:
        print(f"[tokenizer] {message}", flush=True)

    scheme = NumericScheme(
        group_size=args.group_size,
        compound=not args.option_a,
        max_int_digits=args.max_int_digits,
        max_frac_digits=args.max_frac_digits,
    )
    spec, grams, _ = build_tokenizer(
        documents=read_documents(args.corpus, args.limit_bytes),
        scheme=scheme,
        vocab_size=args.vocab_size,
        specials=DEFAULT_SPECIALS,
        min_count=args.min_count,
        max_ngram=args.max_ngram,
        max_trees=args.max_trees,
        pattern=PRETOKEN_PATTERN,
        log=log,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    ngram_path = args.output / spec.ngrams.filename
    digest = ngram_store.write_counts(ngram_path, grams)
    spec = TokenizerSpec(
        numeric=spec.numeric,
        text_tokens=spec.text_tokens,
        ngrams=NgramReference(
            filename=spec.ngrams.filename,
            sha256=digest,
            entries=spec.ngrams.entries,
            min_count=spec.ngrams.min_count,
            max_length=spec.ngrams.max_length,
        ),
        specials=spec.specials,
        pretoken_pattern=spec.pretoken_pattern,
        provenance=spec.provenance,
    )
    spec.write(args.output / "tokenizer.json")
    log(
        f"wrote {args.output}/tokenizer.json "
        f"({spec.vocab_size} tokens, sha256 {spec.sha256()[:16]}) and "
        f"{ngram_path.name} ({ngram_path.stat().st_size / 1e6:.1f} MB)"
    )

    if args.validation:
        tokenizer = SplitTreeNumericTokenizer(spec, grams)
        documents = list(
            read_documents(args.validation, args.validation_limit_bytes)
        )
        stats = evaluate(tokenizer, documents)
        log(
            f"validation: {stats['bytes_per_token']:.4f} bytes/token over "
            f"{stats['documents']:,} documents, "
            f"{stats['roundtrip_failures']} round-trip failures"
        )
        if args.baseline_gpt2:
            baseline = gpt2_baseline(documents)
            stats["gpt2_baseline"] = baseline
            stats["compression_ratio"] = (
                stats["bytes_per_token"] / baseline["bytes_per_token"]
            )
            log(
                f"gpt2 baseline: {baseline['bytes_per_token']:.4f} bytes/token; "
                f"this tokenizer is {stats['compression_ratio']:.3f}x as dense "
                "on the same text"
            )
        (args.output / "validation.json").write_text(
            json.dumps(stats, indent=2) + "\n"
        )
        if stats["roundtrip_failures"]:
            log("FAIL: tokenizer is not invertible on validation text")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
