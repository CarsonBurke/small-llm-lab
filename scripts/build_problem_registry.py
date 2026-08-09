"""Build the global math problem registry and the contamination index.

One command produces both artifacts every builder consults:

* ``registry.parquet`` -- each problem's canonical key and the single split
  that owns it, resolved by the ``eval > rl > sft > pretrain`` priority;
* ``ngrams.npy`` -- the word n-gram index of protected problem statements,
  which catches an evaluation item quoted inside a larger web document.

The output directory is versioned and immutable: the script refuses to write
into a directory that already holds a registry. Build a new version instead.

This reads parquet text only -- no model executes -- so it does not go
through mlq.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Iterator
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postraining.decontaminate import (
    DEFAULT_NGRAM_SIZE,
    ContaminationIndex,
    index_words,
    informative_ngram_hashes,
    strip_framing,
)
from postraining.problem_registry import (
    ProblemRegistry,
    SourceDeclaration,
    load_sources,
    read_problems,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = REPOSITORY_ROOT / "postraining" / "problem_sources.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "problem_registry" / "v1"


def protected_problems(
    sources: list[SourceDeclaration], splits: set[str]
) -> Iterator[str]:
    for source in sources:
        if source.split in splits:
            yield from read_problems(source.path, source.columns)


def detection_control(
    index: ContaminationIndex,
    sources: list[SourceDeclaration],
    splits: set[str],
    per_source: int,
) -> dict:
    """Re-read protected problems and confirm the index actually detects them.

    `index.json` declares which splits it covers, and every consumer trusts
    that declaration. A build that indexed the wrong split, or silently
    dropped one, would produce an index whose manifest is a promise nothing
    checks -- and the failure mode is a corpus that looks decontaminated and
    is not.

    This turns the declaration into a measurement at the one moment the
    source text is still in hand. What it proves is that the indexing pipeline
    covered the problems it was handed -- `strip_framing`, `index_words`,
    `informative`, the rolling hash, and `searchsorted` end to end, against
    text re-read from disk. It cannot detect a later hand-edit of
    `index.json`, because one `splits` value feeds both the build and this
    check.

    Coverage is decided with the builder's own predicate, not a proxy for it.
    A row is uncoverable when it yields no informative n-gram, which happens
    for a row shorter than one window *and* for a long row whose windows are
    all one repeated token -- `Simplify (d*d*((d*d**3)/d)/d)/(d/d**6)` is 18
    words of which 12 are `d`. Testing length instead would count such a row
    as indexable and then blame the index for correctly declining to index it.
    """
    by_split: dict[str, dict[str, int]] = {}
    for source in sources:
        if source.split not in splits:
            continue
        totals = by_split.setdefault(
            source.split, {"indexable": 0, "detected": 0, "uncoverable": 0}
        )
        for seen, problem in enumerate(read_problems(source.path, source.columns)):
            if seen >= per_source:
                break
            # Coverage is decided on the stripped form, because that is what
            # was indexed; detection is queried on the raw form, because that
            # is what a corpus document quoting the problem looks like. A raw
            # problem's windows are a superset of its stripped form's, so the
            # asymmetry can only help detection, never fake it.
            words = index_words(strip_framing(problem))
            if informative_ngram_hashes(words, index.ngram_size).size == 0:
                totals["uncoverable"] += 1
                continue
            totals["indexable"] += 1
            totals["detected"] += index.hit_count(problem) >= 1
    indexable = sum(totals["indexable"] for totals in by_split.values())
    detected = sum(totals["detected"] for totals in by_split.values())
    uncoverable = sum(totals["uncoverable"] for totals in by_split.values())
    if detected != indexable:
        raise SystemExit(
            f"contamination index failed its own positive control: "
            f"{detected}/{indexable} sampled protected problems were detected. "
            f"The indexing pipeline did not cover every problem it was given "
            f"for splits {sorted(splits)}."
        )
    return {
        "sampled_per_source": per_source,
        "indexable": indexable,
        "detected": detected,
        "uncoverable": uncoverable,
        "by_split": by_split,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--index-splits",
        nargs="+",
        default=["eval", "rl", "sft"],
        help=(
            "splits whose problem statements enter the n-gram index. The "
            "default is every split a pretraining builder must refuse: "
            "narrowing it leaves those pools on exact key matching, which for "
            "web text catches only a page that is nothing but a problem. The "
            "index costs 8 bytes per unique n-gram"
        ),
    )
    parser.add_argument("--ngram-size", type=int, default=DEFAULT_NGRAM_SIZE)
    parser.add_argument(
        "--control-samples",
        type=int,
        default=100,
        help=(
            "protected problems per source re-queried against the finished "
            "index. The build fails if any indexable one is not detected, "
            "which is what makes the index's declared split coverage a "
            "measurement rather than a claim"
        ),
    )
    args = parser.parse_args()

    if (args.output / "registry.parquet").exists():
        raise SystemExit(
            f"{args.output} already holds a registry; registries are "
            "immutable, so build a new version directory instead"
        )

    sources = load_sources(args.sources, args.root)
    missing = [source.path for source in sources if not source.path.exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} declared source(s) are missing, first: {missing[0]}"
        )

    registry = ProblemRegistry.build(sources, log=print)
    registry.provenance["sources_declaration"] = {
        "path": str(args.sources),
        "sha256": hashlib.sha256(args.sources.read_bytes()).hexdigest(),
    }
    counts = registry.provenance["split_counts"]
    print(
        f"\nregistry: {len(registry):,} distinct problems "
        f"({', '.join(f'{split} {counts[split]:,}' for split in sorted(counts))})"
    )
    if registry.provenance["conflicts"]:
        print("sources that yielded problems to a higher-priority split:")
        for name, lost in sorted(
            registry.provenance["conflicts"].items(), key=lambda kv: -kv[1]
        ):
            print(f"  {name:40s} {lost:>9,}")

    splits = set(args.index_splits)
    unknown = splits - {source.split for source in sources}
    if unknown:
        raise SystemExit(f"no declared source has split(s) {sorted(unknown)}")
    index = ContaminationIndex.build(
        protected_problems(sources, splits),
        ngram_size=args.ngram_size,
        splits=splits,
    )
    control = detection_control(index, sources, splits, args.control_samples)
    index.provenance["detection_control"] = control

    # Nothing is written until the control passes. Writing the registry first
    # would leave a directory holding half a version -- and the immutability
    # guard at the top of `main` then refuses the retry into it, so a failed
    # gate would burn a version number instead of reporting a problem.
    digest = registry.write(args.output)
    index_digest = index.write(args.output)
    print(
        f"registry sha256 {digest[:16]}\n"
        f"index: {len(index):,} unique {args.ngram_size}-grams from "
        f"{index.covered_rows:,} of {index.provenance['protected_rows']:,} rows "
        f"({index.short_rows:,} yielding no informative n-gram, "
        f"{index.provenance['low_diversity_ngrams']:,} n-grams too repetitive) "
        f"sha256 {index_digest[:16]}"
    )
    print(
        f"positive control: {control['detected']:,}/{control['indexable']:,} "
        f"sampled protected problems detected "
        f"({control['uncoverable']:,} uncoverable by n-gram, protected by "
        "exact key only)"
    )
    for split, totals in sorted(control["by_split"].items()):
        print(
            f"  {split:6s} indexable {totals['indexable']:>6,}  "
            f"detected {totals['detected']:>6,}  "
            f"uncoverable {totals['uncoverable']:>6,}"
        )

    # A copy of the declaration, byte-identical to the input, plus its digest
    # in the registry manifest -- so "which sources produced this registry" is
    # answerable from the artifact alone.
    (args.output / "sources.json").write_bytes(args.sources.read_bytes())
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
