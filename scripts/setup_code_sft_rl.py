#!/usr/bin/env python3
"""Plan one verified code-SFT epoch; execute the plan only as an mlq job.

The following RL stage has 50 critic warmup + 950 actor updates.
RL is configured separately once the selected objective family is known.
"""
from __future__ import annotations

import argparse
import json
import hashlib
from dataclasses import replace
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from itertools import islice
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BASE = "postraining/runs/kda8_sft_omi2_drills_code30k_science_v1_e1/sft_final_model.pt"
POOL = "postraining/data/ultradata-code-l3-v3-sft-pool.parquet"
POOL_MANIFEST = "postraining/data/ultradata-code-l3-v3.manifest.json"
MIXTURE = "postraining/data/reviewed_candidate_20260926_v2/mixture.manifest.json"
SEPARATION_SCHEMA = "code_sft_actual_rl_separation/v1"
SEPARATION_RULE = "word5 MinHash126, LSH63 bands, estimated Jaccard>=0.3; " \
    "LSH is probabilistic and can miss near duplicates"


def stage_plan(name: str) -> dict:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("name must be one directory component")
    traces = f"postraining/data/{name}_traces.parquet"
    python = sys.executable
    prepare = [
        python,
        str(REPO / "scripts/setup_code_sft_rl.py"),
        "--name",
        name,
        "--plan",
        f"postraining/data/{name}.plan.json",
        "--prepare-corpus",
    ]
    train = [
        python,
        "-m",
        "postraining.sft_trace_train",
        "--name",
        name,
        "--checkpoint",
        BASE,
        "--traces",
        traces,
        "--epochs",
        "1",
        "--seq-len",
        "1024",
        "--rows-per-micro-batch",
        "8",
        "--grad-accum",
        "4",
        "--lr-scale",
        "0.1",
        "--warmup-steps",
        "20",
        "--think-tokens",
        "--answer-fence",
        "--holdout-problems",
        "256",
        "--eval-every",
        "20",
        "--skip-sampling-gate",
    ]
    return {
        "schema": "code_sft_before_rl_plan/v1",
        "name": name,
        "base_checkpoint": BASE,
        "verified_pool": POOL,
        "pool_manifest": POOL_MANIFEST,
        "sft_epochs": 1,
        "sft_updates": "determined by packed one-epoch corpus",
        "critic_warmup_updates": 50,
        "rl_actor_updates": 950,
        "rl_total_updates": 1000,
        "rl_mixture": MIXTURE,
        "packed_rows_per_update": 32,
        "max_input_tokens_per_update": 32768,
        "commands": [prepare, train],
        "sft_checkpoint": f"postraining/runs/{name}/sft_final_model.pt",
        "validation": "held-out code completion CE every 20 updates; "
        "execution reward must be measured in the subsequent RL stage",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def prepared_corpus_paths(name: str) -> tuple[Path, Path, Path, Path]:
    stage_plan(name)  # Apply the same name validation to direct callers.
    filtered = REPO / f"postraining/data/{name}_disjoint_pool.parquet"
    traces = REPO / f"postraining/data/{name}_traces.parquet"
    return filtered, filtered.with_suffix(".manifest.json"), traces, traces.with_suffix(".manifest.json")


def reuse_prepared_corpus(name: str, *, filtered_only: bool = False) -> bool:
    """Reuse only a complete corpus whose provenance still matches its inputs."""
    paths = prepared_corpus_paths(name)
    if not any(path.exists() for path in paths):
        return False
    if not all(path.is_file() for path in paths[:2]):
        raise ValueError("partial prepared corpus; refusing to overwrite or reuse it")
    filtered, audit_path, traces, manifest_path = paths
    has_traces = traces.exists() or manifest_path.exists()
    if has_traces and not (traces.is_file() and manifest_path.is_file()):
        raise ValueError("partial prepared corpus; refusing to overwrite or reuse it")

    def read(path: Path) -> dict:
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid preparation manifest: {path}") from error
        if not isinstance(value, dict):
            raise ValueError(f"invalid preparation manifest: {path}")
        return value

    def require(actual, expected, label: str) -> None:
        if actual != expected:
            raise ValueError(f"prepared corpus mismatch: {label}")

    pool_manifest = read(REPO / POOL_MANIFEST)
    mixture = read(REPO / MIXTURE)
    audit = read(audit_path)
    require(audit.get("schema"), SEPARATION_SCHEMA, "separation schema")
    for field, expected in (
        ("source_pool", POOL),
        ("source_pool_manifest", POOL_MANIFEST),
        ("rl_mixture", MIXTURE),
        ("near_duplicate_rule", SEPARATION_RULE),
    ):
        require(audit.get(field), expected, field)
    pool_hash = sha256(REPO / POOL)
    require(pool_hash, pool_manifest.get("sft_pool", {}).get("sha256"), "verified pool hash")
    require(audit.get("source_pool_sha256"), pool_hash, "source pool hash")
    require(audit.get("source_pool_manifest_sha256"), sha256(REPO / POOL_MANIFEST), "source manifest hash")
    require(audit.get("rl_mixture_sha256"), sha256(REPO / MIXTURE), "RL mixture hash")
    for field in ("dataset", "revision"):
        if not pool_manifest.get(field):
            raise ValueError(f"verified pool manifest is missing {field}")
        require(audit.get(field), pool_manifest[field], field)
    sources = []
    for source in mixture["sources"]:
        actual_hash = sha256(REPO / source["path"])
        require(actual_hash, source["sha256"], f"RL source hash: {source['path']}")
        sources.append({key: source[key] for key in ("name", "path", "sha256", "rows")})
    require(audit.get("rl_sources"), sources, "RL source provenance")
    filtered_hash = sha256(filtered)
    require(audit.get("output_sha256"), filtered_hash, "disjoint pool hash")
    import pyarrow.parquet as pq

    pool_rows = pq.ParquetFile(REPO / POOL).metadata.num_rows
    filtered_rows = pq.ParquetFile(filtered).metadata.num_rows
    require(audit.get("input_rows"), pool_rows, "audit input rows")
    require(audit.get("kept_rows"), filtered_rows, "audit kept rows")
    require(audit.get("removed_union"), pool_rows - filtered_rows, "audit removed rows")
    if not 0 < filtered_rows <= pool_rows:
        raise ValueError("prepared corpus mismatch: filtered row counts")
    if not has_traces:
        return filtered_only
    manifest = read(manifest_path)
    require(manifest.get("rl_separation_audit"), str(audit_path.relative_to(REPO)), "separation audit path")
    require(manifest.get("rl_separation_audit_sha256"), sha256(audit_path), "separation audit hash")
    require(manifest.get("output_sha256"), sha256(traces), "trace corpus hash")
    from postraining.math_prompt import ANSWER_FENCE_PROMPT_SCHEMA

    require(manifest.get("schema"), "instruction_corpus_answer_fence/v2", "trace schema")
    require(manifest.get("answer_fence_prompt_schema"), ANSWER_FENCE_PROMPT_SCHEMA, "answer fence schema")
    require(manifest.get("seq_len"), 1024, "sequence length")
    require(manifest.get("verified"), True, "verified trace corpus")
    corpora = manifest.get("corpora")
    if not isinstance(corpora, list) or len(corpora) != 1:
        raise ValueError("prepared corpus mismatch: trace source count")
    for key, expected in (
        ("name", "ultradata_code_l3"),
        ("dataset", pool_manifest["dataset"]),
        ("revision", pool_manifest["revision"]),
    ):
        require(corpora[0].get(key), expected, f"trace source {key}")
    require(manifest.get("shard_sha256"), [{
        "corpus": "ultradata_code_l3", "shard": filtered.name,
        "sha256": filtered_hash,
    }], "trace input shard")
    trace_rows = pq.ParquetFile(traces).metadata.num_rows
    if not 0 < trace_rows <= filtered_rows <= pool_rows:
        raise ValueError("prepared corpus mismatch: corpus row counts")
    require(manifest.get("documents"), trace_rows, "trace document count")
    require(manifest.get("verified_documents"), trace_rows, "verified document count")
    return True


def sft_execution_command(plan: dict) -> list[str]:
    """Continue the trainer's exact saved schedule, keeping the one-epoch budget."""
    command = list(plan["commands"][1])
    run_dir = REPO / "postraining/runs" / plan["name"]
    if not run_dir.exists():
        return command
    if (run_dir / "result.json").exists():
        raise ValueError(f"SFT run already finished: {run_dir}")
    # sft_trace_train binds args, input/source hashes, packed schedule,
    # optimizer states and RNG; it refuses any incompatible resume.
    if not (run_dir / "resume.pt").is_file():
        raise ValueError(f"partial SFT run has no exact resume state: {run_dir}")
    return [*command, "--resume"]


def _signature_batch(texts):
    """Compute unchanged signatures in a worker, returning one compact array."""
    import numpy as np
    from postraining.prepare_ultradata_code import minhash

    return np.stack([minhash(text) for text in texts])


def ordered_signatures(texts, *, workers=4, batch_size=256):
    """Four-process MinHash with bounded queued batches and stable row order."""
    import numpy as np
    from postraining.prepare_ultradata_code import MINHASH_PERMUTATIONS

    if workers < 1 or batch_size < 1:
        raise ValueError("workers and batch_size must be positive")
    signatures = np.empty((len(texts), MINHASH_PERMUTATIONS), dtype=np.uint32)
    batches = iter(texts)
    if not len(texts):
        return signatures
    # Only one batch per worker is outstanding. Copy each completed batch
    # into its final destination instead of retaining all batch results.
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn")
    ) as executor:
        pending = deque()
        for _ in range(workers):
            batch = list(islice(batches, batch_size))
            if not batch:
                break
            pending.append(executor.submit(_signature_batch, batch))
        offset = 0
        while pending:
            result = pending.popleft().result()
            signatures[offset : offset + len(result)] = result
            offset += len(result)
            del result
            batch = list(islice(batches, batch_size))
            if batch:
                pending.append(executor.submit(_signature_batch, batch))
    return signatures


def rl_audit_inputs(path):
    """Read only prompt text and code entry points, in bounded Arrow batches."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    columns = ["prompt"]
    if "verification_info" in parquet.schema_arrow.names:
        columns.append("verification_info.entry_points")
    problems, identities = [], set()
    for batch in parquet.iter_batches(batch_size=512, columns=columns):
        for row in batch.to_pylist():
            problems.append(row["prompt"][-1]["content"])
            entry_points = (row.get("verification_info") or {}).get("entry_points")
            if entry_points:
                identities.add(tuple(sorted(name.lower() for name in entry_points)))
    return problems, identities


def prepare_corpus(name: str) -> None:
    """Exclude overlap against the actual full RL mixture before framing."""
    if reuse_prepared_corpus(name):
        print(f"Reusing hash-verified prepared corpus for {name}", flush=True)
        return
    if reuse_prepared_corpus(name, filtered_only=True):
        print(f"Reusing hash-verified filtered pool for {name}; continuing framing", flush=True)
        filtered, audit_path, _, _ = prepared_corpus_paths(name)
        frame_corpus(name, filtered, audit_path)
        return
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from postraining.prepare_ultradata_code import (
        near_twins,
        normalized_task,
        sft_identity,
    )

    pool_manifest = json.loads((REPO / POOL_MANIFEST).read_text())
    if sha256(REPO / POOL) != pool_manifest["sft_pool"]["sha256"]:
        raise ValueError("verified SFT pool hash differs from its manifest")
    mixture = json.loads((REPO / MIXTURE).read_text())
    rl_problems = []
    rl_entry_ids = set()
    sources = []
    for source in mixture["sources"]:
        path = REPO / source["path"]
        actual_hash = sha256(path)
        if actual_hash != source["sha256"]:
            raise ValueError(f"RL source hash differs: {path}")
        problems, identities = rl_audit_inputs(path)
        rl_problems.extend(problems)
        rl_entry_ids.update(identities)
        del problems, identities
        sources.append(
            {
                "name": source["name"],
                "path": source["path"],
                "sha256": actual_hash,
                "rows": source["rows"],
            }
        )
    rl_exact = {normalized_task(x) for x in rl_problems}
    rl_prefixes = {sft_identity(x) for x in rl_problems}
    pool_columns = pq.read_table(REPO / POOL, columns=["problem", "entry_points"])
    pool_problems = pool_columns.column("problem").to_pylist()
    pool_entries = pool_columns.column("entry_points").to_pylist()
    del pool_columns
    input_rows = len(pool_problems)
    exact = np.array(
        [
            normalized_task(problem) in rl_exact or sft_identity(problem) in rl_prefixes
            for problem in pool_problems
        ]
    )
    identity = np.array(
        [
            tuple(sorted(name.lower() for name in entries)) in rl_entry_ids
            for entries in pool_entries
        ]
    )
    print(
        f"Checking near-duplicate separation: {input_rows} SFT / {len(rl_problems)} RL tasks",
        flush=True,
    )
    query_signatures = ordered_signatures(pool_problems)
    index_signatures = ordered_signatures(rl_problems)
    twins = near_twins(query_signatures, index_signatures)
    del query_signatures, index_signatures
    del pool_problems, pool_entries, rl_problems, rl_exact, rl_prefixes, rl_entry_ids
    keep = ~(exact | identity | twins)
    filtered = REPO / f"postraining/data/{name}_disjoint_pool.parquet"
    audit_path = filtered.with_suffix(".manifest.json")
    if filtered.exists() or audit_path.exists():
        raise FileExistsError(f"refusing to overwrite {filtered}")
    pool = pq.read_table(REPO / POOL)
    pq.write_table(pool.filter(pa.array(keep)), filtered)
    del pool
    audit = {
        "schema": SEPARATION_SCHEMA,
        "source_pool": POOL,
        "source_pool_sha256": sha256(REPO / POOL),
        "source_pool_manifest": POOL_MANIFEST,
        "source_pool_manifest_sha256": sha256(REPO / POOL_MANIFEST),
        "dataset": pool_manifest["dataset"],
        "revision": pool_manifest["revision"],
        "rl_mixture": MIXTURE,
        "rl_mixture_sha256": sha256(REPO / MIXTURE),
        "rl_sources": sources,
        "input_rows": input_rows,
        "kept_rows": int(keep.sum()),
        "exact_or_prefix_overlaps": int(exact.sum()),
        "entry_identity_overlaps": int(identity.sum()),
        "near_duplicate_overlaps": int(twins.sum()),
        "removed_union": int((~keep).sum()),
        "near_duplicate_rule": SEPARATION_RULE,
        "output_sha256": sha256(filtered),
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    frame_corpus(name, filtered, audit_path)


def frame_corpus(name: str, filtered: Path, audit_path: Path) -> None:
    """Frame an already verified separation stage without recomputing it."""
    from postraining import prepare_sft_corpus as prepare

    prepare.ADAPTERS["ultradata_code_l3"] = replace(
        prepare.ADAPTERS["ultradata_code_l3"], local_parquet=filtered
    )
    sys.argv = [
        "prepare_sft_corpus",
        "--source",
        "ultradata_code_l3",
        "--output",
        f"postraining/data/{name}_traces.parquet",
        "--seq-len",
        "1024",
        "--workers",
        "4",
    ]
    prepare.main()
    traces_manifest = REPO / f"postraining/data/{name}_traces.manifest.json"
    manifest = json.loads(traces_manifest.read_text())
    manifest["rl_separation_audit"] = str(audit_path.relative_to(REPO))
    manifest["rl_separation_audit_sha256"] = sha256(audit_path)
    traces_manifest.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prepare-corpus", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="prepare corpus and train; submit this mode through mlq",
    )
    args = parser.parse_args()
    plan = stage_plan(args.name)
    if args.prepare_corpus:
        try:
            prepare_corpus(args.name)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        return
    if args.execute:
        if not args.plan.is_file() or json.loads(args.plan.read_text()) != plan:
            parser.error("execution requires the exact reviewed plan file")
        for required in (BASE, POOL, POOL_MANIFEST):
            if not (REPO / required).is_file():
                parser.error(f"missing required artifact: {required}")
        try:
            train_command = sft_execution_command(plan)
            reuse = reuse_prepared_corpus(args.name)
            if "--resume" in train_command and not reuse:
                raise ValueError("exact SFT resume requires its complete verified corpus")
        except (ValueError, OSError) as error:
            parser.error(str(error))
        if not reuse:
            command = list(plan["commands"][0])
            command[command.index("--plan") + 1] = str(args.plan.resolve())
            subprocess.run(command, cwd=REPO, check=True)
            if not reuse_prepared_corpus(args.name):
                parser.error("preparation did not produce a complete verified corpus")
        subprocess.run(train_command, cwd=REPO, check=True)
        print(f"SFT complete: {plan['sft_checkpoint']}", flush=True)
    else:
        if args.plan.exists():
            parser.error(f"refusing to overwrite {args.plan}")
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        args.plan.write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
