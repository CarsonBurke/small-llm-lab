"""Pinned UltraData source adaptation for the shared verifiable-task pipeline.

Raw byte windows are a bounded, source-order-biased candidate pool; indexed QA
is read separately. No source questions or executable tests are truncated.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import random
import re
from urllib.parse import quote
from postraining.data_acquisition import (
    AcquisitionError,
    BoundedHTTP,
    BoundedParquetFile,
    canonical_json,
)

DATASET = "openbmb/UltraData-RL-2609"
REVISION = "e6ecfa733708a4c54b5a98c3ca0fd16fc6923790"
PUBLISHED_COUNTS = {
    "Math": 32412,
    "Code": 23665,
    "Knowledge": 11872,
    "Long_Context": 18046,
}
SHARD_COUNTS = {"Math": 4, "Code": 12, "Knowledge": 2, "Long_Context": 2}
RECORD_CAP = 256 * 1024
WINDOW_BYTES = 4 * 1024**2
TREE_URL = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}?recursive=true&expand=false"


@dataclass(frozen=True)
class Window:
    path: str
    domain: str
    size: int
    start: int
    length: int
    stratum_start: int
    stratum_end: int


def source_tree(http: BoundedHTTP) -> tuple[list[dict], dict]:
    data, receipt = http.fetch(TREE_URL, limit=1024**2)
    tree = json.loads(data)
    if not isinstance(tree, list):
        raise AcquisitionError("unexpected HF tree metadata")
    shards = []
    counts = Counter()
    for item in tree:
        match = re.fullmatch(
            r"data/(Math|Code|Knowledge|Long_Context)/[^/]+\.jsonl",
            item.get("path", ""),
        )
        if match and item.get("type") == "file":
            domain = match[1]
            if not isinstance(item.get("size"), int) or item["size"] <= 0:
                raise AcquisitionError("invalid shard size")
            shards.append({**item, "domain": domain})
            counts[domain] += 1
    if dict(counts) != SHARD_COUNTS:
        raise AcquisitionError(f"pinned tree shard counts differ: {dict(counts)}")
    return sorted(shards, key=lambda item: item["path"]), receipt


def plan_windows(
    shards: list[dict], *, seed: int, byte_budget: int, extended_prefixes: bool = False
) -> list[Window]:
    """Seeded random strata plus explicitly nonrepresentative shard prefixes.

    Prefixes recover compact tasks where random bytes land in giant records.
    Their inclusion weight is one: observed usable counts, never extrapolated.
    """
    acquisition_budget = byte_budget
    if extended_prefixes:
        # Preserve prior random windows and cache keys when extending acquisition.
        byte_budget = min(byte_budget, 254 * 1024**2)
    small = [item for item in shards if item["size"] <= 12 * 1024**2]
    large = [item for item in shards if item not in small]
    prefix_bytes = sum(
        min(item["size"], RECORD_CAP if item["domain"] == "Code" else 1024**2)
        for item in large
    )
    remaining = byte_budget - sum(item["size"] for item in small) - prefix_bytes
    if large and remaining < len(large) * 2 * RECORD_CAP * 2:
        raise ValueError("network budget too small to cover two strata of every shard")
    windows = [
        Window(
            item["path"], item["domain"], item["size"], 0, item["size"], 0, item["size"]
        )
        for item in small
    ]
    if large:
        rounds = max(2, remaining // (len(large) * WINDOW_BYTES))
        width = min(WINDOW_BYTES, remaining // (len(large) * rounds))
        for item in large:
            rng = random.Random(f"{seed}:{item['path']}")
            for index in range(rounds):
                left, right = (
                    item["size"] * index // rounds,
                    item["size"] * (index + 1) // rounds,
                )
                length = min(width, right - left)
                start = rng.randint(left, right - length)
                windows.append(
                    Window(
                        item["path"],
                        item["domain"],
                        item["size"],
                        start,
                        length,
                        left,
                        right,
                    )
                )
        prefixes = [
            Window(
                item["path"],
                item["domain"],
                item["size"],
                0,
                min(item["size"], RECORD_CAP if item["domain"] == "Code" else 1024**2),
                0,
                min(item["size"], RECORD_CAP if item["domain"] == "Code" else 1024**2),
            )
            for item in large
        ]
        if sum(window.length for window in windows + prefixes) > byte_budget:
            raise ValueError(
                "network budget cannot cover stratified windows and bounded prefixes"
            )
        windows.extend(prefixes)
    if extended_prefixes:
        extensions = []
        for window in windows:
            if window.start != 0 or window.domain not in ("Code", "Long_Context"):
                continue
            end = min(window.size, (16 if window.domain == "Code" else 64) * 1024**2)
            if end > window.length:
                extensions.append(
                    Window(
                        window.path,
                        window.domain,
                        window.size,
                        window.length,
                        end - window.length,
                        window.length,
                        end,
                    )
                )
        windows.extend(extensions)
        if sum(window.length for window in windows) > acquisition_budget:
            raise ValueError("network budget cannot cover extended prefixes")
    return sorted(windows, key=lambda window: (window.path, window.start))


def complete_records(
    payload: bytes, window: Window, counters: Counter, *, record_cap: int = RECORD_CAP
):
    """Yield complete JSONL records and absolute offsets; never join fragments."""
    position = 0
    if window.start:
        first = payload.find(b"\n")
        counters["leading_boundary_bytes"] += len(payload) if first < 0 else first + 1
        if first < 0:
            return
        position = first + 1
    while position < len(payload):
        newline = payload.find(b"\n", position)
        if newline < 0:
            if window.start + len(payload) != window.size:
                counters["trailing_boundary_bytes"] += len(payload) - position
                return
            end = len(payload)
        else:
            end = newline + 1
        length = end - position
        counters["complete_records"] += 1
        if length > record_cap:
            counters["oversized_records"] += 1
        else:
            try:
                value = json.loads(payload[position:end])
            except (ValueError, UnicodeDecodeError):
                counters["invalid_json"] += 1
            else:
                yield value, window.start + position, length
        position = end


_UNVERIFIABLE = re.compile(
    r"\b(prove that|show that|give a proof|justify your answer|explain why|true or false|select all|multiple[ -]choice|which of the following|which options?|refer to (?:the )?(?:figure|image|diagram)|shown in (?:the )?(?:figure|image|diagram))\b|证明|多选|单选|如图",
    re.I,
)
_PROOF_REQUEST = re.compile(
    r"^[^\S\n]*(?:prove\b|verify[ \t]+(?:that|the[ \t]+(?:identity|inequality))\b|demonstrate[ \t]+that\b)",
    re.I | re.M,
)
# Consume indentation only at line starts. Fixed-width inline boundaries and
# newline-free indentation avoid rescanning long whitespace runs quadratically.
_OPTIONS = re.compile(
    r"(?:^[ \t]*(?:[-*][ \t]+)?|(?<=[ \t]{2}))\(?[a-d](?:[.):][ \t]*|[ \t]*-[ \t]+)",
    re.M | re.I,
)
_PARTS = re.compile(r"^[^\S\n]*\(?[a-c1-3][.)]\s+", re.M)
_PAREN_CHOICE_LABELS = re.compile(r"(?<!\w)\(([A-J])\)(?=[\s}\\*])")
_TEX_CHOICE_LABELS = re.compile(
    r"\\(?:text(?:bf|rm)?|mathrm|mathbf)[ \t]*\{[ \t]*\(([A-J])\)"
)


def math_content(text: str) -> str:
    from postraining.problem_registry import strip_framing

    # Unlike registry.normalize_problem: never casefold math variables.
    return re.sub(r"\s+", " ", strip_framing(text)).strip()


def reviewed_questions() -> tuple[list[tuple[str, dict]], str]:
    from postraining.core import _math_corpus_policy

    entries, identity = _math_corpus_policy()
    return [
        (
            math_content("\n".join(message["content"] for message in entry["prompt"])),
            entry,
        )
        for entry in entries.values()
    ], identity


def rejection_reason(
    source: dict,
    domain: str,
    reviews: list[tuple[str, dict]],
    *,
    test_cap: int = RECORD_CAP,
) -> str | None:
    from postraining.verifiable_tasks import (
        MAX_CASE_BYTES,
        MAX_TEST_BYTES,
        MAX_TEST_CASES,
    )

    if not isinstance(source, dict) or source.get("domain") != domain:
        return "invalid_domain"
    if any(
        not isinstance(source.get(key), str) or not source[key].strip()
        for key in ("uuid", "query", "source")
    ):
        return "invalid_source_fields"
    query, target = source["query"], source.get("ground_truth")
    if domain == "Code":
        if not isinstance(target, dict) or set(target) - {
            "inputs",
            "outputs",
            "call_type",
            "fn_name",
        }:
            return "invalid_code_contract"
        inputs, outputs = target.get("inputs"), target.get("outputs")
        if target.get("call_type", "std") != "std" or target.get("fn_name") is not None:
            return "unsupported_code_signature"
        if (
            not isinstance(inputs, list)
            or not isinstance(outputs, list)
            or not inputs
            or len(inputs) != len(outputs)
            or any(not isinstance(value, str) for value in inputs + outputs)
        ):
            return "invalid_code_tests"
        if any(not value.strip() for value in outputs):
            return "empty_code_expected_output"
        sizes = [len(value.encode("utf-8")) for value in inputs + outputs]
        if (
            len(inputs) > MAX_TEST_CASES
            or max(sizes) > MAX_CASE_BYTES
            or sum(sizes) > MAX_TEST_BYTES
        ):
            return "code_verifier_payload_cap"
        if len(canonical_json(target)) > test_cap:
            return "oversized_code_tests"
        return None
    if (
        not isinstance(target, str)
        or not target.strip()
        or len(target.encode("utf-8")) > 4096
    ):
        return "invalid_text_target"
    question = query
    if domain == "Long_Context":
        # Article lists and words such as "prove" are evidence, not task format.
        boundaries = list(re.finditer(r"(?im)^(?:#{1,6}\s*)?Question\s*:\s*", query))
        if boundaries:
            question = query[boundaries[-1].end() :]
    if (
        _UNVERIFIABLE.search(question)
        or _PROOF_REQUEST.search(question)
        or len(_OPTIONS.findall(question)) >= 2
        or len(_PARTS.findall(question)) >= 2
        or (
            "(A)" in question
            and (
                # Four plain alternatives or three explicitly formatted labels;
                # function arguments and three named premises remain valid.
                {"A", "B", "C", "D"}.issubset(_PAREN_CHOICE_LABELS.findall(question))
                or {"A", "B", "C"}.issubset(_TEX_CHOICE_LABELS.findall(question))
            )
        )
    ):
        return "unverifiable_question_structure"
    if target.strip().casefold() in {
        "true",
        "false",
        "yes",
        "no",
        "n/a",
        "unknown",
        "not enough information",
        "unanswerable",
    } or re.fullmatch(r"\(?[A-D]\)?", target.strip()):
        return "unverifiable_text_target"
    if "\n" in target.strip() or len(target.split()) > 48:
        return "non_atomic_text_target"
    if domain == "Knowledge":
        from postraining.core import parse_numeric_answer

        if parse_numeric_answer(target) is None and (
            len(target.split()) > 8
            or re.match(r"(?i)^(?:it|they|this|these|there)\b", target.strip())
        ):
            return "free_form_target_requires_semantic_judge"
    if domain == "Math":
        content = math_content(query)
        for question, entry in reviews:
            if question and question in content:
                if entry["action"] == "quarantine" or target != entry.get(
                    "corrected_target"
                ):
                    return "reviewed_math_quarantine"
    return None


CONVERSION_REVISION = "a9fcbd481b7f80a884c3727102be394329b1f4cc"


def typed_ultradata_source(source: dict, *, revision: str, provenance: dict) -> dict:
    """Translate source-specific grading semantics into an explicit task kind."""
    from postraining.core import parse_numeric_answer

    code = source["domain"] == "Code"
    kind = (
        "python_stdio"
        if code
        else (
            "math"
            if source["domain"] == "Math"
            or parse_numeric_answer(source["ground_truth"]) is not None
            else "text"
        )
    )
    target = source["ground_truth"]
    if code:
        target = {"call_type": "std", "fn_name": None, **target}
    return {
        **source,
        "ground_truth": target,
        "verification_kind": kind,
        "source": DATASET,
        "source_revision": revision,
        "provenance": provenance,
    }


def acquire_ultradata(
    http: BoundedHTTP, *, seed: int = 1337
) -> tuple[list[dict], dict]:
    """Reuse bounded raw discovery; cover indexed QA rather than byte fragments."""
    reviews, review_identity = reviewed_questions()
    shards, tree_receipt = source_tree(http)
    windows = plan_windows(
        shards, seed=seed, byte_budget=1022 * 1024**2, extended_prefixes=True
    )
    counters = {domain: Counter() for domain in PUBLISHED_COUNTS}
    rows, receipts = [], []
    for window in windows:
        if window.domain == "Long_Context":
            continue
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{quote(window.path, safe='/')}"
        payload, receipt = http.fetch(
            url, limit=window.length, start=window.start, total=window.size
        )
        receipts.append({**window.__dict__, **receipt})
        for source, offset, length in complete_records(
            payload, window, counters[window.domain]
        ):
            reason = rejection_reason(source, window.domain, reviews)
            if reason:
                counters[window.domain][reason] += 1
                continue
            rows.append(
                typed_ultradata_source(
                    source,
                    revision=REVISION,
                    provenance={
                        "path": window.path,
                        "byte_offset": offset,
                        "record_bytes": length,
                        "acquisition": "bounded_raw_window",
                    },
                )
            )
            counters[window.domain]["source_eligible"] += 1

    tree_url = f"https://huggingface.co/api/datasets/{DATASET}/tree/{CONVERSION_REVISION}?recursive=true&expand=false"
    payload, converted_tree_receipt = http.fetch(tree_url, limit=1024**2)
    qa_files = sorted(
        (
            entry
            for entry in json.loads(payload)
            if entry.get("type") == "file"
            and re.fullmatch(
                r"Long-Context/train/\d{4}\.parquet", entry.get("path", "")
            )
        ),
        key=lambda entry: entry["path"],
    )
    if len(qa_files) != 8:
        raise AcquisitionError("pinned QA conversion must contain eight parquet files")
    qa_receipts, converted_rows = [], 0
    for entry in qa_files:
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{CONVERSION_REVISION}/{quote(entry['path'], safe='/')}"
        parquet = BoundedParquetFile(http, url, entry["size"])
        converted_rows += parquet.metadata.num_rows
        for group in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(
                group, ["uuid", "query", "ground_truth", "source", "domain"]
            )
            for index, source in enumerate(table.to_pylist()):
                counters["Long_Context"]["complete_records"] += 1
                if (
                    not isinstance(source.get("query"), str)
                    or len(source["query"].encode("utf-8")) > RECORD_CAP
                ):
                    counters["Long_Context"]["oversized_query"] += 1
                    continue
                reason = rejection_reason(source, "Long_Context", reviews)
                if reason:
                    counters["Long_Context"][reason] += 1
                    continue
                rows.append(
                    typed_ultradata_source(
                        source,
                        revision=CONVERSION_REVISION,
                        provenance={
                            "path": entry["path"],
                            "row_group": group,
                            "row_in_group": index,
                            "raw_release_revision": REVISION,
                            "acquisition": "indexed_qa_conversion",
                        },
                    )
                )
                counters["Long_Context"]["source_eligible"] += 1
        qa_receipts.append(
            {
                "path": entry["path"],
                "size": entry["size"],
                "rows": parquet.metadata.num_rows,
                "receipts": parquet.receipts,
            }
        )
        print(
            json.dumps(
                {
                    "event": "qa_source_file",
                    "path": entry["path"],
                    "rows_seen": converted_rows,
                    "filters": dict(counters["Long_Context"]),
                }
            ),
            flush=True,
        )
    if converted_rows != 18045:
        raise AcquisitionError(
            f"pinned QA conversion row count changed: {converted_rows}"
        )
    return rows, {
        "dataset": DATASET,
        "revision": REVISION,
        "conversion_revision": CONVERSION_REVISION,
        "published_counts": PUBLISHED_COUNTS,
        "tree_receipt": tree_receipt,
        "raw_windows": receipts,
        "qa_tree_receipt": converted_tree_receipt,
        "qa_files": qa_receipts,
        "qa_conversion_rows": converted_rows,
        "math_review_identity": review_identity,
        "filters": {domain: dict(counts) for domain, counts in counters.items()},
        "license": "Apache-2.0 dataset card; upstream MIT/CC-BY/CC-BY-SA terms and redistribution restriction remain",
        "limitations": [
            "Raw code is bounded, prefix-biased supplementary coverage; CodeContests supplies broader code coverage.",
            "Pinned QA conversion contains18045 rows versus18046 advertised; no fabricated missing row.",
            "Knowledge prose targets requiring semantic judgement are excluded from exact-answer rewards.",
        ],
    }
