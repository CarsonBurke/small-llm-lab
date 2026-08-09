"""Migrate immutable SFT5 traces to the bare-problem prompt contract.

Only the prompt prefix changes. Every completion byte, row order, label, and
source assignment remains identical to SFT5, making this a controlled prompt
schema migration rather than a new trace-selection experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from postraining.core import THINK_OPEN, GPT2BPETokenizer
from postraining.math_prompt import (
    ANSWER_FENCE_PROMPT_SCHEMA,
    LEGACY_ANSWER_FENCE_PROMPT_SCHEMA,
    LEGACY_ANSWER_FENCE_SUFFIX,
    strip_math_prompt_framing,
)


MIGRATION_SCHEMA = "verified_math_sft_bare_prompt_migration/v1"
DEFAULT_PARENT = Path(
    "postraining/data/sft_traces_v5_answer_canonical_a1swap10k.parquet"
)
DEFAULT_PARENT_SHA256 = (
    "078a8041cbc1c4907be281d6327f006fffa39144ff6b1294582b596a779ed46e"
)
DEFAULT_OUTPUT = Path(
    "postraining/data/sft_traces_v6_answer_bare_a1swap10k.parquet"
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary(path: Path) -> Path:
    return path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp"
    )


def _publish_no_replace(temporary: Path, final: Path) -> None:
    os.link(temporary, final)
    temporary.unlink()


def require_fresh_outputs(output: Path) -> Path:
    manifest = output.with_suffix(".manifest.json")
    existing = [path for path in (output, manifest) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite immutable SFT6 artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    return manifest


def migrate_row(row: dict, tokenizer) -> dict:
    required = {
        "source",
        "problem",
        "document",
        "final_answer",
        "verified",
        "doc_tokens",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"SFT5 row lacks required columns {sorted(missing)}")
    if row["verified"] is not True:
        raise ValueError(
            "SFT6 migration requires every parent row to be verified"
        )

    problem, removed = strip_math_prompt_framing(str(row["problem"]))
    if removed or problem != str(row["problem"]).strip():
        raise ValueError("SFT5 problem column is not already a bare problem")
    legacy_prefix = problem + LEGACY_ANSWER_FENCE_SUFFIX
    document = str(row["document"])
    expected_start = legacy_prefix + THINK_OPEN
    if not document.startswith(expected_start):
        raise ValueError(
            "SFT5 document does not begin with its problem, legacy prompt "
            f"contract, and {THINK_OPEN}: {document[:160]!r}"
        )

    completion = document[len(legacy_prefix):]
    migrated_document = problem + completion
    migrated = dict(row)
    migrated["problem"] = problem
    migrated["document"] = migrated_document
    migrated["doc_tokens"] = len(tokenizer.encode(migrated_document)) + 1
    return migrated


def migrate_rows(rows: list[dict], tokenizer) -> list[dict]:
    migrated = [migrate_row(row, tokenizer) for row in rows]
    if len({str(row["document"]) for row in migrated}) != len(migrated):
        raise ValueError("bare-prompt migration introduced duplicate documents")
    return migrated


def write_outputs(
    rows: list[dict], output: Path, manifest_path: Path, manifest: dict
) -> dict:
    parquet_temporary = _temporary(output)
    manifest_temporary = _temporary(manifest_path)
    published: list[Path] = []
    try:
        pq.write_table(pa.Table.from_pylist(rows), parquet_temporary)
        complete_manifest = {
            **manifest,
            "output": str(output),
            "output_sha256": file_sha256(parquet_temporary),
        }
        manifest_temporary.write_text(
            json.dumps(complete_manifest, indent=2, sort_keys=True) + "\n"
        )
        _publish_no_replace(parquet_temporary, output)
        published.append(output)
        _publish_no_replace(manifest_temporary, manifest_path)
        published.append(manifest_path)
        return complete_manifest
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        parquet_temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)


def build(args: argparse.Namespace) -> dict:
    parent = Path(args.parent)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = require_fresh_outputs(output)

    parent_sha256 = file_sha256(parent)
    if parent_sha256 != args.parent_sha256:
        raise ValueError(
            f"SFT5 parent hash mismatch: {parent_sha256} != "
            f"{args.parent_sha256}"
        )
    parent_manifest_path = parent.with_suffix(".manifest.json")
    if not parent_manifest_path.is_file():
        raise ValueError(
            f"SFT5 parent manifest is missing: {parent_manifest_path}"
        )
    parent_manifest = json.loads(parent_manifest_path.read_text())
    if (
        parent_manifest.get("answer_fence_prompt_schema")
        != LEGACY_ANSWER_FENCE_PROMPT_SCHEMA
    ):
        raise ValueError(
            "SFT5 parent does not use the expected legacy prompt schema"
        )
    if parent_manifest.get("output_sha256") != parent_sha256:
        raise ValueError("SFT5 parent manifest does not bind the parent parquet")

    parent_rows = pq.read_table(parent).to_pylist()
    tokenizer = GPT2BPETokenizer(think_tokens=True, answer_tokens=True)
    rows = migrate_rows(parent_rows, tokenizer)
    if file_sha256(parent) != parent_sha256:
        raise ValueError("SFT5 parent changed during migration")

    source_counts = dict(
        sorted(Counter(str(row["source"]) for row in rows).items())
    )
    manifest = {
        "schema": MIGRATION_SCHEMA,
        "answer_fence_prompt_schema": ANSWER_FENCE_PROMPT_SCHEMA,
        "parent": {
            "path": str(parent),
            "sha256": parent_sha256,
            "manifest": str(parent_manifest_path),
            "manifest_sha256": file_sha256(parent_manifest_path),
            "answer_fence_prompt_schema": LEGACY_ANSWER_FENCE_PROMPT_SCHEMA,
        },
        "documents": len(rows),
        "source_counts": source_counts,
        "doc_tokens": {
            "total": sum(int(row["doc_tokens"]) for row in rows),
            "maximum": max((int(row["doc_tokens"]) for row in rows), default=0),
        },
        "migration": {
            "removed_prompt_suffix": LEGACY_ANSWER_FENCE_SUFFIX,
            "completion_bytes_preserved": True,
            "row_order_preserved": True,
        },
        "args": vars(args),
    }
    return write_outputs(rows, output, manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", default=str(DEFAULT_PARENT))
    parser.add_argument("--parent-sha256", default=DEFAULT_PARENT_SHA256)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    try:
        manifest = build(args)
    except (FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
