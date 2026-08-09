"""Draw a domain-weighted text sample for tokenizer training.

A tokenizer trained on the wrong mixture spends its vocabulary in the wrong
place, so the sample is drawn from the same pinned sources and the same domain
weights as the corpus the tokenizer will encode. Documents are taken by
least-completed byte fraction, the same interleaving rule the corpus builder
uses, so no source is over-represented at the head of the sample.

The same admission guard runs here as in the corpus builder, for two reasons.
A tokenizer is fitted to its sample, so a held-out problem in the sample buys
the model shorter encodings of exactly the text it will be evaluated on --
quieter than training on it, and just as much a leak. And the sample is only
representative of the corpus if it is drawn from the documents that will
actually survive into it.

This reads text only -- no model executes -- so it does not go through mlq.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postraining.decontaminate import ProblemGuard, load_guard
from scripts import build_k3_pretrain_dataset as base
from scripts.build_problem_registry import DEFAULT_OUTPUT


def source_raw_documents(
    source: dict, remote_root: Path, max_chars: int
) -> Iterator[base.RawDocument]:
    """The source's documents exactly as the corpus builder renders them.

    Whole `RawDocument`s rather than their text, because a consumer that only
    sees text loses `quality_keys` -- the problem statements a QA source knows
    about -- and would then measure a different corpus than the builder does.
    """
    files = base.source_files(source, remote_root)
    kind = source["kind"]
    if kind == "parquet_text":
        # The builder's filtering counters are irrelevant to a text sample, but
        # the same filters must run so the sample matches the real stream.
        documents = base.parquet_documents(
            files, source, max_chars, Counter(), defaultdict(Counter)
        )
    elif kind == "jsonl_text":
        documents = base.jsonl_documents(files, source, max_chars)
    elif kind == "openmath_qa":
        documents = base.openmath_documents(files, source, 0.2)
    elif kind == "deepmind_qa":
        documents = base.deepmind_documents(source, 0.2)
    elif kind == "token_shards":
        return
    else:
        raise ValueError(f"unsupported source kind {kind!r} for text sampling")
    yield from documents


def admitted_documents(
    source: dict,
    remote_root: Path,
    max_chars: int,
    guard: ProblemGuard,
    rejected: Counter,
) -> Iterator[str]:
    for document in source_raw_documents(source, remote_root, max_chars):
        reason = guard.reason(document.text, document.quality_keys)
        if reason is not None:
            rejected[reason] += 1
            continue
        yield document.text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=base.DEFAULT_MANIFEST)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--remote-root", type=Path, default=base.DEFAULT_REMOTE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, default=None)
    parser.add_argument("--bytes", type=int, default=200_000_000)
    parser.add_argument("--validation-bytes", type=int, default=4_000_000)
    parser.add_argument("--max-document-chars", type=int, default=32_768)
    parser.add_argument("--problem-registry", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-ngram-hits", type=int, default=1)
    args = parser.parse_args()

    if not (args.problem_registry / "registry.parquet").exists():
        parser.error(
            f"no problem registry at {args.problem_registry}; build one with "
            "scripts/build_problem_registry.py. Fitting a tokenizer to the "
            "evaluation problems would shorten exactly the text it is judged "
            "on."
        )
    guard, _ = load_guard(
        args.problem_registry, split="pretrain", min_ngram_hits=args.min_ngram_hits
    )
    print(
        f"problem registry: {len(guard.excluded):,} problems excluded, "
        f"{len(guard.index):,} protected {guard.index.ngram_size}-grams",
        flush=True,
    )

    manifest = json.loads(args.manifest.read_text())
    weights = {source["name"]: source["weight"] for source in manifest["sources"]}
    if args.weights is not None:
        profile = json.loads(args.weights.read_text())
        weights = dict(profile["sources"])

    total = args.bytes + args.validation_bytes
    budgets = {
        name: int(round(total * weight / sum(weights.values())))
        for name, weight in weights.items()
        if weight > 0
    }
    streams = {}
    rejected: dict[str, Counter] = defaultdict(Counter)
    for source in manifest["sources"]:
        name = source["name"]
        if name not in budgets:
            continue
        if source["kind"] == "token_shards":
            # Already tokenized under the old vocabulary; unusable as text.
            del budgets[name]
            continue
        streams[name] = admitted_documents(
            source,
            args.remote_root,
            args.max_document_chars,
            guard,
            rejected[name],
        )
    written = {name: 0 for name in streams}
    exhausted: set[str] = set()

    train_handle = args.output.open("w")
    validation_handle = (
        args.validation_output.open("w") if args.validation_output else None
    )
    train_bytes = 0
    validation_bytes = 0
    try:
        while len(exhausted) < len(streams):
            # Least completed byte fraction first, matching the builder.
            name = min(
                (n for n in streams if n not in exhausted),
                key=lambda n: written[n] / budgets[n],
            )
            if written[name] >= budgets[name]:
                exhausted.add(name)
                continue
            text = next(streams[name], None)
            if text is None:
                exhausted.add(name)
                continue
            size = len(text.encode("utf-8"))
            written[name] += size
            record = json.dumps({"text": text, "source": name}) + "\n"
            if (
                validation_handle is not None
                and validation_bytes < args.validation_bytes
            ):
                validation_handle.write(record)
                validation_bytes += size
            else:
                train_handle.write(record)
                train_bytes += size
    finally:
        train_handle.close()
        if validation_handle is not None:
            validation_handle.close()

    print(f"train {train_bytes / 1e6:.1f} MB -> {args.output}")
    if validation_handle is not None:
        print(f"validation {validation_bytes / 1e6:.1f} MB -> {args.validation_output}")
    for name in sorted(written):
        counts = rejected[name]
        detail = (
            "  ".join(f"{reason} {count:,}" for reason, count in sorted(counts.items()))
            if counts
            else "clean"
        )
        print(f"  {name:24s} {written[name] / 1e6:8.1f} MB  rejected {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
