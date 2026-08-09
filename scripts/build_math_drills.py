"""Generate the worked arithmetic drill corpus and its held-out probe panel.

The two artifacts are built together on purpose. A probe is only a held-out
measurement if it is disjoint from the exact training stream it will be used
to judge, and the cheapest way to guarantee that is to derive both from the
same generator in one command: the training keys are computed first, the panel
is drawn with those keys excluded, and the disjointness is then re-verified
independently before anything is written.

The output directory is versioned and immutable; the script refuses to write
over an existing drill set.

This is pure CPU generation -- no model executes -- so it does not go through
mlq.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postraining.arithmetic_probe import (
    PROBE_CURRICULUM,
    PROBE_SEED,
    TRAINING_SEED,
    assert_disjoint,
    build_panel,
    write_panel,
)
from postraining.math_drills import (
    DEFAULT_CURRICULUM,
    DRILL_SCHEMA,
    drill_statistics,
    generate,
)
from postraining.problem_registry import ProblemRegistry

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "math_drills" / "v1"
DEFAULT_REGISTRY = REPOSITORY_ROOT / "data" / "problem_registry" / "v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=TRAINING_SEED)
    parser.add_argument("--probe-seed", type=int, default=PROBE_SEED)
    parser.add_argument("--probe-per-family", type=int, default=128)
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="problem registry whose eval/rl/sft problems drills must avoid; "
        "pass --no-registry only when no registry exists yet",
    )
    parser.add_argument("--no-registry", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(
            f"{args.output} already exists; drill sets are immutable, so "
            "build a new version directory instead"
        )
    if args.seed == args.probe_seed:
        raise SystemExit(
            "the training and probe seeds must differ, or the panel would be "
            "drawn from the training stream itself"
        )

    excluded: frozenset[bytes] = frozenset()
    registry_provenance: dict | None = None
    if not args.no_registry:
        if not (args.registry / "registry.parquet").exists():
            raise SystemExit(
                f"no problem registry at {args.registry}; build one with "
                "scripts/build_problem_registry.py, or pass --no-registry to "
                "generate drills that have not been checked against the "
                "evaluation and post-training splits"
            )
        registry = ProblemRegistry.read(args.registry)
        excluded = frozenset(registry.excluded_from("pretrain"))
        registry_provenance = {
            "directory": str(args.registry),
            "registry_sha256": registry.provenance["registry_sha256"],
            "excluded_problem_keys": len(excluded),
        }
        print(f"registry: excluding {len(excluded):,} claimed problems")

    print(f"generating {args.count:,} drills at seed {args.seed}", flush=True)
    drills = list(
        generate(
            seed=args.seed,
            count=args.count,
            curriculum=DEFAULT_CURRICULUM,
            excluded_keys=excluded,
        )
    )
    statistics = drill_statistics(drills)

    args.output.mkdir(parents=True)
    table = pa.table(
        {
            "family": pa.array([d.family for d in drills], type=pa.string()),
            "digit_order": pa.array(
                [d.digit_order for d in drills], type=pa.string()
            ),
            "digits": pa.array(
                [int(d.difficulty.get("digits", 0)) for d in drills], type=pa.int16()
            ),
            "problem": pa.array([d.problem for d in drills], type=pa.string()),
            "solution": pa.array([d.solution for d in drills], type=pa.string()),
            "answer": pa.array([d.answer for d in drills], type=pa.string()),
            "document": pa.array([d.document() for d in drills], type=pa.string()),
        }
    )
    drills_path = args.output / "drills.parquet"
    pq.write_table(table, drills_path, compression="zstd")
    drills_digest = hashlib.sha256(drills_path.read_bytes()).hexdigest()

    print(
        f"probe: drawing {args.probe_per_family} items per family at seed "
        f"{args.probe_seed}",
        flush=True,
    )
    training = {drill.key for drill in drills}
    panel = build_panel(
        per_family=args.probe_per_family,
        curriculum=PROBE_CURRICULUM,
        seed=args.probe_seed,
        excluded_keys=frozenset(training | excluded),
    )
    # Re-derive the training keys from the generator rather than reusing the
    # set above, so the check cannot pass because of a shared mistake.
    assert_disjoint(
        panel,
        training_seed=args.seed,
        training_count=args.count,
        curriculum=DEFAULT_CURRICULUM,
    )
    probe_path = args.output / "probe.jsonl"
    write_panel(
        panel,
        probe_path,
        {
            "seed": args.probe_seed,
            "per_family": args.probe_per_family,
            "training_seed": args.seed,
            "training_count": args.count,
            "min_digits": min(item.digits for item in panel),
            "disjoint_from_training": True,
        },
    )
    probe_digest = hashlib.sha256(probe_path.read_bytes()).hexdigest()

    manifest = {
        "schema": DRILL_SCHEMA,
        "seed": args.seed,
        "requested_count": args.count,
        "drills_sha256": drills_digest,
        "probe_sha256": probe_digest,
        "probe_items": len(panel),
        "probe_seed": args.probe_seed,
        "problem_registry": registry_provenance,
        "statistics": statistics,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    print(
        f"\n{statistics['drills']:,} drills, "
        f"{statistics['characters'] / 1e6:.1f} MB of text, "
        f"{statistics['mean_document_characters']:.0f} characters each"
    )
    for family, count in statistics["families"].items():
        print(f"  {family:18s} {count:>9,}")
    print(f"digit orders: {statistics['digit_orders']}")
    print(f"probe: {len(panel):,} items, disjoint from training")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
