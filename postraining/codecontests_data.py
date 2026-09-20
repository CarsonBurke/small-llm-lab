"""Pinned, train-only CodeContests acquisition with intact hidden stdio suites.

Only projected Parquet columns are fetched. This is a conservative exact-stdout
subset, not a reimplementation of the original contest judges. Descriptions can
omit checker requirements; the manifest records that remaining source limitation.
"""

from __future__ import annotations

from collections import Counter
import json
import random
import re

from postraining.data_acquisition import (
    AcquisitionError,
    BoundedHTTP,
    BoundedParquetFile,
    canonical_json,
    sha256,
)
from postraining.verifiable_tasks import MAX_CASE_BYTES, MAX_TEST_BYTES, MAX_TEST_CASES

DATASET = "deepmind/code_contests"
REVISION = "802411c3010cb00d1b05bad57ca77365a3c699d6"
LICENSE = "CC-BY-4.0"
TRAIN_SHARDS = 39
PUBLISHED_TRAIN_ROWS = 13328
TREE_URL = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}?recursive=true&expand=false"
CARD_URL = f"https://huggingface.co/datasets/{DATASET}/raw/{REVISION}/README.md"
SUITES = ("public_tests", "private_tests", "generated_tests")
SOURCES = (
    "UNKNOWN_SOURCE",
    "CODECHEF",
    "CODEFORCES",
    "HACKEREARTH",
    "CODEJAM",
    "ATCODER",
    "AIZU",
)
COLUMNS = (
    "name",
    "description",
    "source",
    *SUITES,
    "input_file",
    "output_file",
    "cf_contest_id",
    "cf_index",
    "cf_tags",
    "cf_rating",
    "difficulty",
    "is_description_translated",
    "time_limit",
    "memory_limit_bytes",
)
_TRAIN_FILE = re.compile(r"data/train-(\d{5})-of-00039-[0-9a-f]{16}\.parquet")
_INTERACTIVE = re.compile(r"\binteractive\b|\binteractor\b", re.I)
_CUSTOM_CHECKER = re.compile(
    r"\b(?:special|custom|output[- ]only)\s+(?:judge|checker|problem)\b"
    r"|\b(?:absolute|relative)\s+(?:or\s+(?:absolute|relative)\s+)?error\b"
    r"|\b(?:error|tolerance)\s+(?:of|does not exceed|not exceeding|at most)\b"
    r"|\b(?:within|up to)\s+(?:an?\s+)?(?:accuracy|precision)\b",
    re.I,
)
_MULTIPLE_OUTPUTS = re.compile(
    r"\b(?:print|output|return)\s+(?:any|an? arbitrary)\b"
    r"|\b(?:in\s+)?any\s+order\b"
    r"|\b(?:multiple|several|more than one)\s+(?:correct|valid|possible)\s+(?:answers?|outputs?|solutions?)\b"
    r"|\b(?:if|when)\s+(?:there (?:are|is)\s+)?(?:multiple|several|many|more than one)\s+(?:answers?|outputs?|solutions?)\b"
    r"|\bany\s+(?:correct|valid)\s+(?:answer|output|solution)\b",
    re.I,
)
_FILE_IO = re.compile(
    r"\b(?:input|output)\s+file\s*:\s*(?!standard\b|stdin\b|stdout\b)\S+"
    r"|\b(?:read|write|reading|writing)\b[^\n.]{0,60}\b[\w-]+\.(?:in|out)\b",
    re.I,
)


def select_train_shards(tree: list[dict], *, seed: int, max_shards: int) -> list[dict]:
    """One seeded shard from each contiguous index stratum; never sample rows."""
    if type(max_shards) is not int or not 1 <= max_shards <= TRAIN_SHARDS:
        raise ValueError(f"max_shards must be between 1 and {TRAIN_SHARDS}")
    if not isinstance(tree, list):
        raise AcquisitionError("CodeContests tree must be a list")
    train = {}
    for entry in tree:
        if not isinstance(entry, dict):
            raise AcquisitionError("invalid CodeContests tree entry")
        path = entry.get("path", "")
        if not isinstance(path, str):
            raise AcquisitionError("invalid CodeContests source path")
        if not path.startswith("data/train-"):
            continue
        match = _TRAIN_FILE.fullmatch(path)
        if match is None or entry.get("type") != "file":
            raise AcquisitionError(f"unexpected pinned train file: {path}")
        index = int(match.group(1))
        size = entry.get("size")
        if index in train or type(size) is not int or size < 12:
            raise AcquisitionError(f"invalid/duplicate pinned train file: {path}")
        train[index] = entry
    if set(train) != set(range(TRAIN_SHARDS)):
        raise AcquisitionError(
            "pinned CodeContests tree must contain all 39 train shards"
        )
    rng = random.Random(seed)
    return [
        train[
            rng.randrange(
                i * TRAIN_SHARDS // max_shards, (i + 1) * TRAIN_SHARDS // max_shards
            )
        ]
        for i in range(max_shards)
    ]


def normalize_codecontests_row(
    row: dict, *, file: str, row_group: int, row_index: int, file_row_index: int
) -> dict:
    """Reject a whole unsupported problem; never repair, trim or sample its tests."""
    name, description = row.get("name"), row.get("description")
    if (
        not isinstance(name, str)
        or not name.strip()
        or not isinstance(description, str)
        or not description.strip()
    ):
        raise ValueError("missing_question")
    source = row.get("source")
    if type(source) is not int or not 0 <= source < len(SOURCES):
        raise ValueError("invalid_source_identity")
    for field in ("input_file", "output_file"):
        value = row.get(field)
        if not isinstance(value, str):
            raise ValueError("missing_io_contract")
        if value.strip():
            raise ValueError("non_stdio")
    tags = row.get("cf_tags")
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise ValueError("invalid_format_metadata")
    text = name + "\n" + description + "\n" + " ".join(tags)
    for pattern, reason in (
        (_INTERACTIVE, "interactive"),
        (_CUSTOM_CHECKER, "custom_checker_or_tolerance"),
        (_MULTIPLE_OUTPUTS, "multiple_valid_outputs"),
        (_FILE_IO, "non_stdio"),
    ):
        if pattern.search(text):
            raise ValueError(reason)

    inputs, outputs = [], []
    suite_counts = {}
    total_bytes = 0
    for suite_name in SUITES:
        suite = row.get(suite_name)
        if not isinstance(suite, dict):
            raise ValueError(f"missing_{suite_name}")
        stdin, expected = suite.get("input"), suite.get("output")
        if (
            not isinstance(stdin, list)
            or not isinstance(expected, list)
            or len(stdin) != len(expected)
        ):
            raise ValueError(f"invalid_{suite_name}")
        suite_counts[suite_name] = len(stdin)
        if len(inputs) + len(stdin) > MAX_TEST_CASES:
            raise ValueError("too_many_test_cases")
        for input_text, output_text in zip(stdin, expected):
            if (
                not isinstance(input_text, str)
                or not isinstance(output_text, str)
                or not output_text.strip()
            ):
                raise ValueError(f"invalid_{suite_name}")
            try:
                sizes = (
                    len(input_text.encode("utf-8")),
                    len(output_text.encode("utf-8")),
                )
            except UnicodeEncodeError as exc:
                raise ValueError(f"invalid_{suite_name}") from exc
            if max(sizes) > MAX_CASE_BYTES:
                raise ValueError("test_case_bytes_exceeded")
            total_bytes += sum(sizes)
            if total_bytes > MAX_TEST_BYTES:
                raise ValueError("test_suite_bytes_exceeded")
        inputs.extend(stdin)
        outputs.extend(expected)
    if not suite_counts["private_tests"] + suite_counts["generated_tests"]:
        raise ValueError("public_only_tests")

    identity = {
        "dataset": DATASET,
        "revision": REVISION,
        "file": file,
        "row_group": row_group,
        "row_index": row_index,
    }
    provenance = {
        **identity,
        "split": "train",
        "file_row_index": file_row_index,
        "license": LICENSE,
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "name": name,
        "original_source_id": source,
        "original_source": SOURCES[source],
        "cf_contest_id": row.get("cf_contest_id"),
        "cf_index": row.get("cf_index"),
        "cf_rating": row.get("cf_rating"),
        "difficulty": row.get("difficulty"),
        "cf_tags": tags,
        "is_description_translated": row.get("is_description_translated"),
        "time_limit": row.get("time_limit"),
        "memory_limit_bytes": row.get("memory_limit_bytes"),
        "suite_counts": suite_counts,
        "test_bytes": total_bytes,
        "full_suite_sha256": sha256(
            canonical_json({suite: row[suite] for suite in SUITES})
        ),
        "checker_policy": "conservative_exact_stdout_subset_not_original_judge",
    }
    return {
        "uuid": "codecontests:" + sha256(canonical_json(identity)),
        "query": name + "\n\n" + description,
        "ground_truth": {
            "call_type": "std",
            "fn_name": None,
            "inputs": inputs,
            "outputs": outputs,
        },
        "source": DATASET,
        "domain": "Code",
        "source_revision": REVISION,
        "verification_kind": "python_stdio",
        "provenance": provenance,
    }


def acquire_codecontests(
    http: BoundedHTTP, *, seed: int = 1337, max_shards: int = 39
) -> tuple[list[dict], dict]:
    """Acquire all usable rows of selected train shards, within the caller's cap.

    Acquisition failures propagate: a failed column/group is never presented as
    a complete source. Prompt token budgets, deduplication and train/dev splitting
    belong to the source-agnostic corpus builder, not this adapter.
    """
    tree_bytes, tree_receipt = http.fetch(TREE_URL, limit=1024**2)
    try:
        tree = json.loads(tree_bytes)
    except (ValueError, UnicodeDecodeError) as exc:
        raise AcquisitionError("invalid CodeContests tree JSON") from exc
    selected = select_train_shards(tree, seed=seed, max_shards=max_shards)
    card_bytes, card_receipt = http.fetch(CARD_URL, limit=1024**2)
    if not re.search(rb"(?im)^\s*-\s*cc-by-4\.0\s*$", card_bytes):
        raise AcquisitionError("pinned CodeContests card does not confirm CC-BY-4.0")

    rows, files = [], []
    counters, source_counts = Counter(), Counter()
    for entry in selected:
        path = entry["path"]
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{path}"
        parquet = BoundedParquetFile(http, url, entry["size"])
        metadata = parquet.metadata
        available = set(metadata.schema.to_arrow_schema().names)
        if not set(COLUMNS) <= available:
            raise AcquisitionError(
                f"missing projected CodeContests columns: {sorted(set(COLUMNS) - available)}"
            )
        file_info = {
            "path": path,
            "url": url,
            "size": entry["size"],
            "oid": entry.get("oid"),
            "lfs": entry.get("lfs"),
            "split": "train",
            "rows": metadata.num_rows,
            "row_groups": [],
            "projected_columns": list(COLUMNS),
        }
        offset = 0
        for group_index in range(metadata.num_row_groups):
            group = metadata.row_group(group_index)
            selected_columns = [
                group.column(i)
                for i in range(group.num_columns)
                if group.column(i).path_in_schema.split(".", 1)[0] in COLUMNS
            ]
            table = parquet.read_row_group(group_index, columns=list(COLUMNS))
            if table.num_rows != group.num_rows:
                raise AcquisitionError(
                    f"partial CodeContests row group: {path}:{group_index}"
                )
            accepted_before = len(rows)
            group_rejected = Counter()
            for index, raw in enumerate(table.to_pylist()):
                counters["examined_rows"] += 1
                try:
                    normalized = normalize_codecontests_row(
                        raw,
                        file=path,
                        row_group=group_index,
                        row_index=index,
                        file_row_index=offset + index,
                    )
                except ValueError as exc:
                    reason = str(exc)
                    counters[reason] += 1
                    group_rejected[reason] += 1
                    continue
                rows.append(normalized)
                counters["accepted_rows"] += 1
                source_counts[normalized["provenance"]["original_source"]] += 1
            file_info["row_groups"].append(
                {
                    "index": group_index,
                    "file_row_start": offset,
                    "rows": group.num_rows,
                    "accepted_rows": len(rows) - accepted_before,
                    "filter_counts": dict(sorted(group_rejected.items())),
                    "projected_compressed_bytes": sum(
                        column.total_compressed_size for column in selected_columns
                    ),
                    "projected_uncompressed_bytes": sum(
                        column.total_uncompressed_size for column in selected_columns
                    ),
                }
            )
            offset += group.num_rows
        if offset != metadata.num_rows:
            raise AcquisitionError(f"inconsistent CodeContests file row count: {path}")
        file_info["download_receipts"] = list(parquet.receipts)
        files.append(file_info)

    manifest = {
        "schema": "codecontests_acquisition/v1",
        "dataset": DATASET,
        "revision": REVISION,
        "license": LICENSE,
        "license_card": CARD_URL,
        "attribution": "DeepMind CodeContests / Li et al., Competition-Level Code Generation with AlphaCode (2022); original sources retained per row.",
        "split": "train",
        "seed": seed,
        "available_train_shards": TRAIN_SHARDS,
        "selected_train_shards": len(selected),
        "published_train_rows": PUBLISHED_TRAIN_ROWS,
        "selected_source_rows": sum(file["rows"] for file in files),
        "accepted_rows": len(rows),
        "filter_counts": dict(sorted(counters.items())),
        "original_source_counts": dict(sorted(source_counts.items())),
        "selection": "one seeded random shard per contiguous shard-index stratum; all rows examined",
        "source_files": files,
        "tree_receipt": tree_receipt,
        "card_receipt": card_receipt,
        "verifier_bounds": {
            "max_test_cases": MAX_TEST_CASES,
            "max_case_bytes": MAX_CASE_BYTES,
            "max_test_bytes": MAX_TEST_BYTES,
        },
        "coverage_exclusions": [
            "Validation and test splits are never read.",
            "Unselected training shards are not represented; shard-index strata do not guarantee difficulty or source balance.",
            "Whole problems with incomplete/malformed suites, public-only tests, oversized full suites, or identifiable unsupported judges are excluded.",
            "No correct/incorrect solution columns are downloaded, and no reference-solution preflight is required.",
            "Prompt budgets and global deduplication are applied later; accepted_rows is not a final unique usable-task count.",
        ],
        "quality_limitations": [
            "Scraped descriptions and supplied expected outputs are not independently certified; generated tests inherit source/reference-solution limitations.",
            "The dataset has no universal checker-type field. Text/tag/file filters conservatively exclude identifiable interactive, file-I/O, multiple-answer and tolerance-based problems, but cannot certify every remaining problem has unique exact stdout.",
            "All public, private and generated arrays are preserved in order, outside the query; natural-language descriptions may already contain public examples.",
            "Observed full-suite bounds can bias retained problem difficulty; no hidden tests are removed to make a problem pass the bounds.",
        ],
    }
    return rows, manifest
