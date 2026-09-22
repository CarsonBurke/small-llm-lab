#!/usr/bin/env python3
"""Build a deterministic DeepMind/UltraData math corpus for MiniCPM RL."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from postraining.core import load_unique_math_rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    audit: dict = {}
    rows = load_unique_math_rows(path, audit=audit)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepmind-source", type=Path, required=True)
    parser.add_argument("--ultra-source", type=Path, required=True)
    parser.add_argument(
        "--base-source", type=Path,
        help="existing mixed corpus whose non-Math rows and Math row count are retained",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ultra-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--rows", type=int, default=0,
        help="total rows to retain; zero keeps all Ultra rows and matches its ratio",
    )
    args = parser.parse_args()
    if not 0.0 < args.ultra_fraction < 1.0:
        parser.error("--ultra-fraction must be between zero and one")
    if args.rows < 0:
        parser.error("--rows must be nonnegative")

    deepmind = read_rows(args.deepmind_source)
    ultra_all = read_rows(args.ultra_source)
    ultra = [
        row for row in ultra_all
        if (row.get("extra_info") or {}).get("domain") == "Math"
    ]
    if not deepmind or not ultra:
        parser.error("both sources must contain at least one math row")

    rng = random.Random(args.seed)
    rng.shuffle(deepmind)
    rng.shuffle(ultra)
    base_nonmath: list[dict] = []
    base_math_count: int | None = None
    if args.base_source is not None:
        base_rows = read_rows(args.base_source)
        base_math = [
            row for row in base_rows
            if (row.get("extra_info") or {}).get("domain") == "Math"
        ]
        base_nonmath = [
            row for row in base_rows
            if (row.get("extra_info") or {}).get("domain") != "Math"
        ]
        base_math_count = len(base_math)
    target_math_rows = args.rows or base_math_count
    if target_math_rows:
        ultra_n = min(len(ultra), round(target_math_rows * args.ultra_fraction))
        deepmind_n = min(len(deepmind), target_math_rows - ultra_n)
        if deepmind_n + ultra_n < target_math_rows:
            parser.error("requested rows exceed available source rows")
        ultra = ultra[:ultra_n]
        deepmind = deepmind[:deepmind_n]
    else:
        ultra_n = len(ultra)
        deepmind_n = round(ultra_n * (1.0 - args.ultra_fraction) / args.ultra_fraction)
        deepmind = deepmind[:min(len(deepmind), deepmind_n)]

    rows = base_nonmath + deepmind + ultra
    rng.shuffle(rows)
    if not rows:
        parser.error("selected corpus is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    temporary.replace(args.output)
    manifest = {
        "schema": "minicpm_math_mix/v1",
        "seed": args.seed,
        "ultra_fraction_requested": args.ultra_fraction,
        "rows": len(rows),
        "base_source": None if args.base_source is None else str(args.base_source),
        "base_nonmath_rows": len(base_nonmath),
        "source_rows": {"deepmind_math": len(deepmind), "ultradata_math": len(ultra)},
        "source_sha256": {
            "deepmind_math": sha256(args.deepmind_source),
            "ultradata": sha256(args.ultra_source),
            **({} if args.base_source is None else {"base": sha256(args.base_source)}),
        },
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
