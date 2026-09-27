#!/usr/bin/env python3
"""Bind audited math corpora into one manifest, checking every retained pair.

CPU data preparation: submit through mlq. This writes no training artifacts and
does not copy the component Parquet files.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def completed_bundle(directory: Path, paths: list[Path], target_bytes: int) -> dict:
    manifest_path = directory / "manifest.json"
    complete = json.loads((directory / "COMPLETE.json").read_text())
    if (sha256(manifest_path) != complete["manifest_sha256"]
            or sha256(directory / "audit.json") != complete["audit_sha256"]):
        raise ValueError("completed bundle hashes differ")
    manifest = json.loads(manifest_path.read_text())
    if ([Path(item["manifest"]).resolve() for item in manifest["components"]] != paths
            or manifest["target_bytes"] != target_bytes):
        raise ValueError("resume parameters differ from completed bundle")
    for item in manifest["components"]:
        component = Path(item["manifest"])
        if (sha256(component) != item["sha256"]
                or sha256(component.parent / "audit.json") != item["audit_sha256"]):
            raise ValueError("component manifest changed since bundle completion")
    for item in manifest["outputs"]:
        path = Path(item["path"])
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError("output changed since bundle completion")
    protection = manifest["protection"]
    for reference in protection["references"]:
        if sha256(Path(reference["path"])) != reference["sha256"]:
            raise ValueError("protected reference changed since bundle completion")
    if sha256(Path(protection["rl_manifest"])) != protection["rl_manifest_sha256"]:
        raise ValueError("RL manifest changed since bundle completion")
    return manifest


def bundle(manifest_paths: list[Path], output: Path, target_bytes: int,
           resume: bool = False) -> dict:
    if not manifest_paths or target_bytes <= 0:
        raise ValueError("components and a positive target are required")
    staging = output.with_name(output.name + ".building")
    paths = [path.resolve() for path in manifest_paths]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate component manifest")
    if output.exists():
        if resume:
            return completed_bundle(output, paths, target_bytes)
        raise FileExistsError("use a fresh bundle destination or --resume")
    if staging.exists():
        if not resume:
            raise FileExistsError("interrupted bundle publication; use --resume")
        if staging.is_symlink():
            raise ValueError("refusing a symlink staging directory")
        if (staging / "COMPLETE.json").exists():
            result = completed_bundle(staging, paths, target_bytes)
            staging.rename(output)
            return result
        # The expensive data is in immutable component shards. An unfinished
        # report can safely be regenerated; never remove unfamiliar files.
        leftovers = list(staging.iterdir())
        allowed = {name + suffix for name in ("audit.json", "manifest.json", "COMPLETE.json")
                   for suffix in ("", ".tmp")}
        if any(p.name not in allowed
               or not p.is_file() or p.is_symlink() for p in leftovers):
            raise ValueError("unknown file in interrupted bundle staging")
        for path in leftovers:
            path.unlink()
        staging.rmdir()
    components, outputs = [], []
    protected = None
    pairs, questions, seen_paths = set(), set(), set()
    total_bytes = total_rows = 0
    source_counts = Counter()
    for path in paths:
        digest = sha256(path)
        manifest = json.loads(path.read_text())
        complete = json.loads((path.parent / "COMPLETE.json").read_text())
        if digest != complete["manifest_sha256"]:
            raise ValueError(f"component completion mismatch: {path}")
        audit_path = path.parent / "audit.json"
        if sha256(audit_path) != complete["audit_sha256"]:
            raise ValueError(f"component audit completion mismatch: {path}")
        audit = json.loads(audit_path.read_text())
        if audit != manifest["audit"] or audit.get("protected_question_matches") != 0:
            raise ValueError(f"component has no valid zero-overlap audit: {path}")
        protection = manifest["protection"]
        if protected is None:
            protected = protection
        elif protection != protected:
            raise ValueError("components use different protected references or matching policies")
        component_rows = component_bytes = 0
        listed = manifest["outputs"]
        if {entry["path"] for entry in listed} != {p.name for p in path.parent.glob("*.parquet")}:
            raise ValueError("component Parquet inventory differs from its manifest")
        for entry in listed:
            shard = (path.parent / entry["path"]).resolve()
            if (shard in seen_paths or shard.stat().st_size != entry["bytes"]
                    or sha256(shard) != entry["sha256"]):
                raise ValueError(f"duplicate or changed shard: {shard}")
            seen_paths.add(shard)
            rows = raw_bytes = 0
            for batch in pq.ParquetFile(shard).iter_batches(
                batch_size=256, columns=["problem", "generated_solution"]
            ):
                for row in batch.to_pylist():
                    problem = row["problem"].encode()
                    solution = row["generated_solution"].encode()
                    pair = hashlib.sha256(problem + b"\x00" + solution).digest()
                    if pair in pairs:
                        raise ValueError("duplicate question/solution across bundle components")
                    pairs.add(pair)
                    questions.add(hashlib.sha256(problem).digest())
                    rows += 1
                    raw_bytes += len(problem) + len(solution)
            if rows != entry["rows"] or raw_bytes != entry["raw_text_bytes"]:
                raise ValueError(f"shard census differs from manifest: {shard}")
            outputs.append({**entry, "path": str(shard), "component_manifest": str(path)})
            component_rows += rows
            component_bytes += raw_bytes
        if component_rows != audit["rows"] or component_bytes != manifest["raw_text_bytes"]:
            raise ValueError("component totals differ from independent census")
        if sha256(path) != digest:
            raise ValueError("component manifest changed during bundle audit")
        total_rows += component_rows
        total_bytes += component_bytes
        source_counts[manifest["source"]["repo_id"]] += component_rows
        components.append({"manifest": str(path), "sha256": digest,
                           "audit_sha256": complete["audit_sha256"],
                           "source": manifest["source"], "rows": component_rows,
                           "raw_text_bytes": component_bytes})
        print(json.dumps({"phase": "bundle_census", "components": len(components),
                          "rows": total_rows, "raw_text_bytes": total_bytes}), flush=True)
    assert protected is not None
    for reference in protected["references"]:
        if sha256(Path(reference["path"])) != reference["sha256"]:
            raise ValueError("protected reference changed since component preparation")
    if sha256(Path(protected["rl_manifest"])) != protected["rl_manifest_sha256"]:
        raise ValueError("RL manifest changed since component preparation")
    audit = {"schema": "disjoint_math_sft_bundle_audit/v1", "rows": total_rows,
             "unique_questions": len(questions), "raw_text_bytes": total_bytes,
             "question_solution_duplicates": 0, "protected_question_matches": 0,
             "method": "Full component shard SHA-256 and question/solution census; zero-overlap results inherited from completed component rescans with identical frozen references.",
             "limitations": ["Question-level lexical and formatting-aware screening cannot prove absence of semantic paraphrases.",
                             "Unrelated protected questions quoted solely inside solutions are not screened.",
                             "Answer consistency and upstream verification do not prove every derivation correct."]}
    manifest = {"schema": "disjoint_math_sft_bundle/v1", "components": components,
                "outputs": outputs, "protection": protected,
                "source": {"repo_id": "local:disjoint-math-sft-bundle"},
                "counts": {"retained": total_rows}, "unique_questions": len(questions),
                "raw_text_bytes": total_bytes, "compressed_bytes": sum(x["bytes"] for x in outputs),
                "size_definition": "UTF-8 bytes of problem plus generated_solution, counted once",
                "target_bytes": target_bytes, "target_met": total_bytes >= target_bytes,
                "rows_by_repository": dict(source_counts), "audit": audit,
                "builder_sha256": sha256(Path(__file__))}
    staging.mkdir(parents=True)
    write_json(staging / "audit.json", audit)
    write_json(staging / "manifest.json", manifest)
    write_json(staging / "COMPLETE.json", {
        "manifest_sha256": sha256(staging / "manifest.json"),
        "audit_sha256": sha256(staging / "audit.json")})
    staging.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-bytes", type=int, default=50_000_000_000)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse validated completion or restart the audit from saved component shards")
    args = parser.parse_args()
    result = bundle(args.component, args.output_dir, args.target_bytes, args.resume)
    print(json.dumps({"rows": result["counts"]["retained"],
                      "raw_text_bytes": result["raw_text_bytes"],
                      "target_met": result["target_met"]}))


if __name__ == "__main__":
    main()
