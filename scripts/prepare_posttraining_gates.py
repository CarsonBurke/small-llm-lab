"""Build immutable, source-stratified frozen-policy panels from RL pools.

Run through mlq. Panels diagnose learnability, not held-out generalization.
They bind the actual SFT corpus and current reward contracts. No training
decision is implied by producing these files or by reward variance alone.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path

from postraining.core import (
    load_unique_math_rows,
    math_corpus_identity,
    math_corpus_policy_sha256,
)
from postraining.math_prompt import ANSWER_FENCE_PROMPT_SCHEMA
from postraining.prepare_vapo_mixture import SOURCE_SPECS, atomic_json, atomic_parquet
from postraining.vapo.code_reward import PYTHON_REWARD_SCHEMA
from postraining.vapo.mixture import VAPO_MIXTURE_SCHEMA, file_sha256, load_mixture_manifest


def stratum(row: dict, source: str) -> str:
    module = str((row.get("extra_info") or {}).get("module", source))
    # Balance science datasets, not their occasional option-count variants.
    return module.rsplit("_choice_", 1)[0] if source == "science_mc" else module


def select_panel(rows: list[dict], source: str, count: int) -> list[dict]:
    if count < 1 or count > len(rows):
        raise ValueError(f"requested {count} distinct prompts from {len(rows)} rows")
    buckets = defaultdict(list)
    for row in rows:
        buckets[stratum(row, source)].append(row)

    def key(row: dict) -> str:
        prompt = json.dumps(row["prompt"], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(("readiness-v1\0" + prompt).encode()).hexdigest()

    queues = [deque(sorted(buckets[name], key=key)) for name in sorted(buckets)]
    selected = []
    while len(selected) < count:
        for queue in queues:
            if queue and len(selected) < count:
                selected.append(queue.popleft())
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompts", type=int, default=512)
    args = parser.parse_args()
    # A new directory makes partial failure visible and prevents mutation of
    # data underneath a queued/running gate.
    args.output.mkdir(parents=True, exist_ok=False)
    sft_hash = file_sha256(args.sft_corpus)
    for name, path, verifier, default_on in SOURCE_SPECS:
        if not default_on and name not in {"ultradata_code_l3", "science_mc", "ultradata_knowledge"}:
            continue
        rows = load_unique_math_rows(path)
        selected = select_panel(rows, name, args.prompts)
        panel_path = args.output / f"{name}.parquet"
        atomic_parquet(selected, panel_path)
        audit = {}
        effective = load_unique_math_rows(panel_path, audit=audit)
        if len(effective) != args.prompts:
            raise ValueError(f"{name}: panel changed during effective-corpus loading")
        manifest = {
            "schema": VAPO_MIXTURE_SCHEMA,
            "math_corpus_policy_sha256": math_corpus_policy_sha256(),
            "answer_fence_prompt_schema": ANSWER_FENCE_PROMPT_SCHEMA,
            "python_reward_schema": PYTHON_REWARD_SCHEMA,
            "sft_corpus": str(args.sft_corpus),
            "sft_corpus_sha256": sft_hash,
            "prompts_per_cycle": len(effective),
            "sources": [{
                "name": name, "path": str(panel_path), "verifier": verifier,
                "rows": len(effective), "sha256": file_sha256(panel_path),
                "math_corpus_identity": math_corpus_identity(effective),
                "math_corpus_audit": audit,
            }],
            "diagnostic_panel": {
                "selection": "readiness-v1_prompt_sha256_stratum_round_robin",
                "parent_path": str(path), "parent_sha256": file_sha256(path),
                "parent_rows": len(rows),
                "strata": dict(Counter(stratum(row, name) for row in effective)),
                "purpose": "RL training-pool learnability; not a held-out benchmark",
            },
        }
        manifest_path = args.output / f"{name}.manifest.json"
        atomic_json(manifest, manifest_path)
        load_mixture_manifest(manifest_path)
        print(json.dumps({"manifest": str(manifest_path), **manifest["diagnostic_panel"]}))


if __name__ == "__main__":
    main()
