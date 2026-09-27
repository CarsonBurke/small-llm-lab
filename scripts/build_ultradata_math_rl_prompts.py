"""Build the UltraData-RL-2609 math RL prompt pool for nano-scale policies.

``postraining/task_data.py`` acquires UltraData-RL-2609 for the MiniCPM
campaign: it pins ``--context-tokens 10000``, tokenizes with MiniCPM5-1B, and
bundles CodeContests.  None of that transfers to a small GPT-2-vocab policy,
and re-acquiring the release to change a tokenizer would discard the existing
hash-bound extraction for nothing.  This script instead promotes the Math rows
of an extraction into a standalone RL pool.

Only the Math domain is taken.  Measured against GPT-2 BPE, UltraData math
problems have a median of 77 tokens and 97.9% fit a 256-token prompt
budget, while Code sits at a 502-token median (16% fit) and Long_Context at
4036 (0% fit), so those domains cannot be episodes for this policy.

v3 took the whole Math domain; v5 (the default) is the same build after the
verifier v4 change and the ``equation_target`` screen: ``verifiable_mixed_10k_20260917``
converted all 32,412 published Math records and kept the 23,361 with a
deterministic, atomic target (``verifiable_corpus/v1`` source filters), in
both of its splits.  Rows then pass the shared screens of
``postraining.math_rl_pool`` -- canonicalization, control-byte corruption,
target self-verification, evaluation decontamination, the 256-token prompt
budget, and restatement collapse -- each counted, with the affected row ids,
in the manifest.

This pool *owns* every problem it shares with DAPO-Math-17K: the same
competition problems reach both, and where their labels disagree the
UltraData copy was the corrected one in every case inspected (NOTES.md
2026-09-23).  ``scripts/build_dapo_rl_prompts.py`` rebuilds dapo to yield.
Pass ``--exclude`` only to make this pool yield to another one instead.

Rows keep the source ``prompt``/``reward_model``/``verification_info`` schema;
prompt canonicalization is the mixture builder's job.

    python3 scripts/build_ultradata_math_rl_prompts.py
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

DEFAULT_EXTRACTIONS = (Path("postraining/data/verifiable_mixed_10k_20260917"),)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extraction", type=Path, action="append",
        help="ultradata_subset/v1 or verifiable_corpus/v1 directory; "
        "repeatable (default: verifiable_mixed_10k_20260917)",
    )
    parser.add_argument(
        "--exclude", type=Path, action="append", default=[],
        help="pool whose problems this one yields (whitespace, skeleton, or "
        "near-duplicate match); repeatable. Default: none, this pool owns "
        "its overlap with dapo",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("postraining/data/ultradata-math-rl-v5.parquet"),
    )
    args = parser.parse_args()
    extractions = args.extraction or list(DEFAULT_EXTRACTIONS)
    manifest_path = args.output.with_suffix(".manifest.json")
    for path in (args.output, manifest_path):
        if path.exists():
            parser.error(
                f"{path} exists; prompt pools are immutable and hash-bound "
                "into published mixture manifests"
            )
    for path in args.exclude:
        if not path.exists():
            parser.error(f"--exclude {path} does not exist")

    try:
        candidates, counts, provenance = load_ultradata_math(extractions)
    except ValueError as error:
        parser.error(str(error))

    guard = EvaluationGuard()
    budget = PromptBudget()
    owners, owner_skipped = {}, {}
    for path in args.exclude:
        owned = []
        for row in load_unique_math_rows(path):
            try:
                owned.append((str(row["extra_info"].get("index")), canonical_problem(row)))
            except ValueError:
                owner_skipped[str(path)] = owner_skipped.get(str(path), 0) + 1
        owners[str(path)] = owned
    result = screen_math_pool(
        candidates,
        identity=lambda row: row["extra_info"]["original_query_sha256"],
        guard=guard,
        budget=budget,
        owners=owners,
    )
    if not result.kept:
        parser.error("no math rows survived the screens")
    counts.update(result.counts)

    # Deterministic order bound to content, not to input-file order; the
    # loader refuses extractions that disagree on a query's task, so the pool
    # is reproducible from any permutation of --extraction.
    rows = []
    for position, candidate in enumerate(
        sorted(result.kept, key=lambda candidate: candidate.identity)
    ):
        row = candidate.row
        row["extra_info"] = {**row["extra_info"], "pool_position": position}
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_name(args.output.name + f".{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pylist(rows), staging)
    staging.replace(args.output)
    # The pool must load back unchanged through the trainer's own loader: no
    # review-registry edits, no conflicting-contract quarantines, same order.
    reloaded = load_unique_math_rows(args.output)
    if math_corpus_identity(reloaded) != math_corpus_identity(rows):
        args.output.unlink()
        raise SystemExit("written pool does not reload to the screened rows")
    manifest = {
        "schema": MATH_RL_POOL_SCHEMA,
        "problems": len(rows),
        "math_corpus_identity": math_corpus_identity(rows),
        **provenance,
        "counts": dict(counts),
        "prompt_budget": budget.provenance(),
        "prompt_tokens": prompt_token_audit(result.prompt_tokens, budget.budget),
        "decontamination": guard.provenance(),
        "yields_to": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "uncanonicalizable_owner_rows_not_matched": owner_skipped.get(
                    str(path), 0
                ),
            }
            for path in args.exclude
        ],
        "policy": policy_provenance(),
        "near_duplicates": result.near_duplicates,
        "quarantined": result.quarantined,
        "dropped": result.dropped,
        "order": "ascending original_query_sha256; no RNG",
        "deduplicated_by": "extra_info.original_query_sha256, then "
        "postraining.problem_overlap components",
        "prompt_framing": (
            "source MiniCPM5 chat-template framing retained verbatim; the "
            "mixture builder canonicalizes it"
        ),
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
