"""Rebuild the DAPO-Math-17K RL prompt pool, disjoint from UltraData math.

DAPO-Math-17K and UltraData-RL-2609's Math domain draw on the same AoPS/AMC/
AIME/CN-olympiad problems. Whitespace-identical prompts are only a fraction of
what they share: most shared problems differ by LaTeX formatting, a kept or
dropped figure, or DAPO's integer-answer rewrite ("... Please provide the
value of m + n"). UltraData owns the overlap (its labels were the corrected
ones where the copies disagree; NOTES.md 2026-09-23), so this pool yields
every problem that restates an UltraData Math problem under any matcher of
``postraining.problem_overlap``.

It yields to the UltraData Math *candidates* -- every extracted Math query,
before that pool's own screens -- rather than to the screened pool. A problem
UltraData dropped because its restatements carry conflicting labels, or
because it is an evaluation near-match, is no safer under DAPO's label.

The remaining rows pass the same screens as the UltraData pool
(``postraining.math_rl_pool``): DAPO had never been decontaminated against
the evaluation targets or measured against the prompt budget, and its own
restatements were never collapsed.

The output keeps the source's first physical variant of each retained prompt,
in first-appearance order, so ``load_unique_math_rows`` applies the same
target-review receipts to it as to the source; the build checks that the
reloaded pool's ``math_corpus_identity`` equals that of the screened rows.

    python3 scripts/build_dapo_rl_prompts.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postraining.core import load_unique_math_rows, math_corpus_identity
from postraining.math_rl_pool import (
    MATH_RL_POOL_SCHEMA,
    EvaluationGuard,
    PromptBudget,
    canonical_problem,
    file_sha256,
    load_ultradata_math,
    policy_provenance,
    prompt_token_audit,
    screen_math_pool,
)

DEFAULT_SOURCE = Path("postraining/data/dapo-math-17k.parquet")
DEFAULT_OWNER_EXTRACTIONS = (Path("postraining/data/verifiable_mixed_10k_20260917"),)


def prompt_key(prompt: list[dict]) -> str:
    return json.dumps(prompt, ensure_ascii=False, sort_keys=True)


def first_physical_rows(path: Path, keys: set[str]) -> dict[str, dict]:
    """The first physical source row of each prompt in ``keys``."""
    found: dict[str, dict] = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            key = prompt_key(row["prompt"])
            if key in keys and key not in found:
                found[key] = row
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--yield-to-ultradata", type=Path, action="append",
        help="UltraData extraction whose Math problems own any overlap; "
        "repeatable (default: verifiable_mixed_10k_20260917)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("postraining/data/dapo-math-17k-v2-ud3dedup.parquet"),
    )
    args = parser.parse_args()
    extractions = args.yield_to_ultradata or list(DEFAULT_OWNER_EXTRACTIONS)
    manifest_path = args.output.with_suffix(".manifest.json")
    for path in (args.output, manifest_path):
        if path.exists():
            parser.error(
                f"{path} exists; prompt pools are immutable and hash-bound "
                "into published mixture manifests"
            )

    source_audit: dict = {}
    source_rows = load_unique_math_rows(args.source, audit=source_audit)
    try:
        owner_rows, _, owner_provenance = load_ultradata_math(extractions)
    except ValueError as error:
        parser.error(str(error))

    owned, owner_skipped = [], 0
    for row in owner_rows:
        try:
            owned.append(
                (row["extra_info"]["original_query_sha256"], canonical_problem(row))
            )
        except ValueError:
            owner_skipped += 1
    guard = EvaluationGuard()
    budget = PromptBudget()
    positions = {id(row): position for position, row in enumerate(source_rows)}
    result = screen_math_pool(
        source_rows,
        identity=lambda row: f"{positions[id(row)]:08d}",
        guard=guard,
        budget=budget,
        owners={"ultradata_math": owned},
        describe=lambda row: f"{positions[id(row)]:08d}",
    )
    if not result.kept:
        parser.error("no dapo rows survived the screens")
    # Identities are first-appearance positions, so this is source order.
    kept = sorted(result.kept, key=lambda candidate: candidate.identity)
    effective = [candidate.row for candidate in kept]
    raw = first_physical_rows(
        args.source, {prompt_key(row["prompt"]) for row in effective}
    )
    rows = [raw[prompt_key(row["prompt"])] for row in effective]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_name(args.output.name + f".{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pylist(rows), staging)
    staging.replace(args.output)
    identity = math_corpus_identity(effective)
    if math_corpus_identity(load_unique_math_rows(args.output)) != identity:
        args.output.unlink()
        raise SystemExit("written pool does not reload to the screened rows")
    manifest = {
        "schema": MATH_RL_POOL_SCHEMA,
        "problems": len(rows),
        "math_corpus_identity": identity,
        "dataset": "BytedTsinghua-SIA/DAPO-Math-17k",
        "source": {"path": str(args.source), "sha256": file_sha256(args.source)},
        "source_math_corpus_audit": {
            key: value
            for key, value in source_audit.items()
            if not isinstance(value, list)
        },
        "counts": dict(result.counts),
        "prompt_budget": budget.provenance(),
        "prompt_tokens": prompt_token_audit(result.prompt_tokens, budget.budget),
        "decontamination": guard.provenance(),
        "yields_to": {
            "ultradata_math": {
                **owner_provenance,
                "candidates": len(owner_rows),
                "uncanonicalizable_owner_rows_not_matched": owner_skipped,
            }
        },
        "policy": policy_provenance(),
        "near_duplicates": result.near_duplicates,
        "quarantined": result.quarantined,
        "dropped": result.dropped,
        "order": "source first-appearance order; no RNG",
        "rows": "first physical source variant of each retained prompt; "
        "target reviews re-apply on load",
        "output_sha256": file_sha256(args.output),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(
        {
            key: manifest[key]
            for key in ("problems", "counts", "prompt_tokens", "output_sha256")
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
