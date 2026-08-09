"""Measure how much of the pretraining corpus is a post-training problem.

Two independent tests run over every sampled document:

* exact -- the document's whole text, normalized, is a registered problem;
* n-gram -- the document reproduces a protected problem's word n-gram, which
  is what catches an evaluation item quoted inside a larger page.

Results are reported per corpus source and per registry split. A source with
a high exact rate is a QA pool that pretraining should not repeat; a source
with n-gram hits is leaking evaluation text through prose.

Sampling is bounded, so a zero here is evidence about the sampled prefix and
nothing more. The number of bytes each source contributed is printed for
exactly that reason.

This reads text only -- no model executes -- so it does not go through mlq.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collections.abc import Iterator

from postraining.decontaminate import ContaminationIndex, ProblemGuard
from postraining.problem_registry import ProblemRegistry
from scripts import build_k3_pretrain_dataset as base
from scripts.build_problem_registry import DEFAULT_OUTPUT
from scripts.sample_tokenizer_corpus import source_raw_documents


def audited_documents(
    source: dict, remote_root: Path, max_chars: int
) -> Iterator[base.RawDocument]:
    """Every source's documents, including the pre-tokenized ones.

    `token_shards` holds fineweb under the GPT-2 vocabulary rather than as
    text, and the corpus builder decodes it back before applying the guard.
    The audit must decode it too: skipping the largest source in the corpus
    and reporting the rest would be a measurement of the easy part.
    """
    if source["kind"] != "token_shards":
        yield from source_raw_documents(source, remote_root, max_chars)
        return
    from scripts.build_math_mix_dataset import GPT2BatchEncoder

    tokenizer = GPT2BatchEncoder()
    files = base.source_files(source, remote_root)
    batch: list = []
    for tokens in base.token_shard_documents(files[0]):
        batch.append(tokens)
        if len(batch) < 256:
            continue
        yield from _decoded(batch, source, tokenizer)
        batch = []
    if batch:
        yield from _decoded(batch, source, tokenizer)


def _decoded(batch, source: dict, tokenizer) -> Iterator[base.RawDocument]:
    texts = tokenizer.decode(
        [tokens[1:].astype("int64", copy=False).tolist() for tokens in batch]
    )
    for text in texts:
        yield base.RawDocument(
            source=source["name"],
            domain=source["domain"],
            segments=(text,),
            quality_keys=(text,),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=base.DEFAULT_MANIFEST)
    parser.add_argument("--remote-root", type=Path, default=base.DEFAULT_REMOTE_ROOT)
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--bytes-per-source", type=int, default=50_000_000)
    parser.add_argument("--max-document-chars", type=int, default=32_768)
    parser.add_argument("--min-ngram-hits", type=int, default=1)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    registry = ProblemRegistry.read(args.registry)
    index = ContaminationIndex.read(args.registry)
    # The builder's own admission test, not a reimplementation of it. An audit
    # that applied a different rule would answer a question nobody asked.
    guard = ProblemGuard(
        registry, index, split="pretrain", min_ngram_hits=args.min_ngram_hits
    )
    print(
        f"registry {len(registry):,} problems; index {len(index):,} "
        f"{index.ngram_size}-grams over {index.covered_rows:,} rows "
        f"({index.short_rows:,} too short to index)\n"
    )

    manifest = json.loads(args.manifest.read_text())
    selected = set(args.sources) if args.sources else None
    findings: dict[str, dict] = {}
    for source in manifest["sources"]:
        name = source["name"]
        if selected is not None and name not in selected:
            continue
        documents = 0
        sampled_bytes = 0
        rejected = Counter()
        splits = Counter()
        examples: list[str] = []
        for document in audited_documents(
            source, args.remote_root, args.max_document_chars
        ):
            text = document.text
            size = len(text.encode("utf-8"))
            if sampled_bytes + size > args.bytes_per_source and documents:
                break
            documents += 1
            sampled_bytes += size
            reason = guard.reason(text, document.quality_keys)
            if reason is None:
                continue
            rejected[reason] += 1
            if reason == "registry_exact":
                # Which split claimed it, for the per-split breakdown. Whole
                # text first, then the source's own keys, in the guard's order.
                for candidate in (text, *document.quality_keys):
                    split = registry.split_of(candidate)
                    if split is not None:
                        splits[split] += 1
                        break
            elif len(examples) < 3:
                examples.append(text[:300])
        total = sum(rejected.values())
        findings[name] = {
            "documents": documents,
            "sampled_bytes": sampled_bytes,
            "rejected": dict(rejected),
            "rejection_rate": total / max(documents, 1),
            "exact_by_split": dict(splits),
            "ngram_examples": examples,
        }
        print(
            f"{name:24s} {documents:>9,} docs {sampled_bytes / 1e6:>8.1f} MB  "
            f"exact {rejected['registry_exact']:>8,} "
            f"({rejected['registry_exact'] / max(documents, 1):6.2%}) "
            f"{dict(splits) if splits else ''}  "
            f"ngram {rejected['registry_ngram']:>7,} "
            f"({rejected['registry_ngram'] / max(documents, 1):6.2%})"
        )

    if args.report:
        args.report.write_text(
            json.dumps(
                {
                    "registry": str(args.registry),
                    "registry_sha256": registry.provenance["registry_sha256"],
                    "index_sha256": index.provenance["ngrams_sha256"],
                    "bytes_per_source": args.bytes_per_source,
                    "min_ngram_hits": args.min_ngram_hits,
                    "sources": findings,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
