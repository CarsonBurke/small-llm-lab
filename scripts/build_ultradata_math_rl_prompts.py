"""Build the UltraData-RL-2609 math RL prompt pool for nano-scale policies.

``postraining/task_data.py`` acquires UltraData-RL-2609 for the MiniCPM
campaign: it pins ``--context-tokens 10000``, tokenizes with MiniCPM5-1B, and
bundles CodeContests.  None of that transfers to a 1024-context GPT-2-vocab
policy, and re-acquiring 4 GiB to change a tokenizer would discard the
existing hash-bound extraction for nothing.  This script instead promotes the
Math rows of that extraction into a standalone RL pool.

Only the Math domain is taken.  Measured against GPT-2 BPE, UltraData math
prompts have a median of 132 tokens and 95% fit a 256-token prompt budget,
while Code sits at a 502-token median (16% fit) and Long_Context at 4036
(0% fit), so those domains cannot be episodes for this policy.

Rows keep the source ``prompt``/``reward_model``/``verification_info`` schema;
prompt canonicalization is the mixture builder's job.  Rows whose problem body
contains a literal ``Answer:`` cannot be canonicalized without destroying the
problem, so they are quarantined and counted rather than silently reshaped.

    python3 scripts/build_ultradata_math_rl_prompts.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import re

from postraining.math_prompt import answer_fence_prompt


def normalized(text: str) -> str:
    """Whitespace- and case-insensitive prompt identity for cross-pool dedup."""
    return re.sub(r"\s+", " ", text).strip().lower()

# UltraData-RL-2609's Math domain draws on the same upstream pools as
# DAPO-Math-17K: 664 of 2,970 prompts (22.4%) are exact-normalized duplicates
# of dapo-math-17k. Both are separate RL mixture sources with their own
# quotas, so an overlapping prompt would be drawn under two names, counted
# twice against the pool, and split its own learnability telemetry. The
# smaller, newer pool yields.
DEFAULT_EXCLUSIONS = (Path("postraining/data/dapo-math-17k.parquet"),)

DEFAULT_EXTRACTIONS = (
    Path("postraining/data/ultradata_small_20260917"),
    Path("postraining/data/ultradata_small_8k_20260917"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extraction", type=Path, action="append",
        help="ultradata_subset/v1 directory; repeatable (default: both)",
    )
    parser.add_argument(
        "--exclude", type=Path, action="append",
        help="parquet whose prompts must not appear in this pool; repeatable "
        "(default: dapo-math-17k, which overlaps this corpus upstream)",
    )
    parser.add_argument(
        "--no-default-exclusions", action="store_true",
        help="build without excluding any other pool; the result may then "
        "double-count prompts if both pools enter one mixture",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("postraining/data/ultradata-math-rl.parquet"),
    )
    args = parser.parse_args()
    extractions = args.extraction or list(DEFAULT_EXTRACTIONS)
    if args.exclude and args.no_default_exclusions:
        parser.error("--exclude and --no-default-exclusions are exclusive")
    if args.no_default_exclusions:
        exclusions = []
    else:
        exclusions = args.exclude or list(DEFAULT_EXCLUSIONS)
    excluded_prompts: set[str] = set()
    exclusion_hashes = []
    for path in exclusions:
        if not path.exists():
            parser.error(f"--exclude {path} does not exist")
        for row in pq.read_table(path).to_pylist():
            try:
                excluded_prompts.add(
                    normalized(answer_fence_prompt(row["prompt"][0]["content"]))
                )
            except ValueError:
                continue
        exclusion_hashes.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    if args.output.exists():
        parser.error(
            f"{args.output} exists; prompt pools are immutable and this one "
            "is hash-bound into published mixture manifests"
        )

    revisions, schemas, reward_identities = set(), set(), set()
    rows, seen, quarantined = [], set(), []
    counts = {
        "seen": 0,
        "non_math": 0,
        "duplicate": 0,
        "uncanonicalizable": 0,
        "missing_identity": 0,
        "excluded_overlap": 0,
    }
    input_hashes = []

    for directory in extractions:
        manifest = json.loads((directory / "manifest.json").read_text())
        for name in ("manifest.json", "train.parquet", "validation.parquet"):
            digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            input_hashes.append(
                {"path": str(directory / name), "sha256": digest}
            )
        revisions.add(manifest["source_revision"])
        schemas.add(manifest["schema"])
        reward_identities.add(manifest["reward_identity"])
        # Both source splits are RL prompt candidates: the extraction's
        # train/validation cut served its assessment, not this pool, and the
        # RL bench panels are separate held-out files.
        for split in ("train", "validation"):
            for row in pq.read_table(directory / f"{split}.parquet").to_pylist():
                counts["seen"] += 1
                info = row["extra_info"]
                if info.get("domain") != "Math":
                    counts["non_math"] += 1
                    continue
                identity = info.get("original_query_sha256")
                if not identity:
                    # Without a content identity the row cannot be
                    # deduplicated; a None key would collapse every such row
                    # into one "duplicate".
                    counts["missing_identity"] += 1
                    continue
                if identity in seen:
                    counts["duplicate"] += 1
                    continue
                try:
                    canonical = answer_fence_prompt(row["prompt"][0]["content"])
                except ValueError as error:
                    counts["uncanonicalizable"] += 1
                    quarantined.append(
                        {"index": info.get("index"), "reason": str(error)[:200]}
                    )
                    continue
                if normalized(canonical) in excluded_prompts:
                    counts["excluded_overlap"] += 1
                    continue
                seen.add(identity)
                rows.append(row)

    if not rows:
        parser.error("no math rows survived the extraction filters")
    if len(revisions) != 1:
        parser.error(f"extractions disagree on source_revision: {revisions}")
    if schemas != {"ultradata_subset/v1"}:
        parser.error(f"unexpected extraction schema: {schemas}")
    if len(reward_identities) != 1:
        parser.error(f"extractions disagree on reward_identity: {reward_identities}")

    # Deterministic order bound to content, not to input-file order, so the
    # pool is reproducible from any permutation of --extraction.
    rows.sort(key=lambda r: r["extra_info"]["original_query_sha256"])
    for position, row in enumerate(rows):
        row["extra_info"] = {**row["extra_info"], "pool_position": position}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_name(args.output.name + f".{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pylist(rows), staging)
    staging.replace(args.output)
    manifest = {
        "problems": len(rows),
        "dataset": "openbmb/UltraData-RL-2609",
        "source_revision": revisions.pop(),
        "source_schema": "ultradata_subset/v1",
        "reward_identity": reward_identities.pop(),
        "domain": "Math",
        "extractions": [str(path) for path in extractions],
        "input_sha256": input_hashes,
        "excluded_pools": exclusion_hashes,
        "counts": counts,
        "quarantined": quarantined,
        "order": "ascending original_query_sha256; no RNG",
        "deduplicated_by": "extra_info.original_query_sha256",
        "prompt_framing": (
            "source MiniCPM5 chat-template framing retained verbatim; the "
            "mixture builder canonicalizes it"
        ),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "quarantined"}, indent=2))
    print(f"quarantined {len(quarantined)} row(s)")


if __name__ == "__main__":
    main()
