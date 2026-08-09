"""Build an immutable, answer-only DAPO/DeepMind/GSM8K OPSD mixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Hashable

import pyarrow as pa
import pyarrow.parquet as pq

from postraining.core import (
    GPT2BPETokenizer,
    encode_prompt,
    normalize_final_answer,
    parse_numeric_answer,
    verify_answer,
)
from postraining.math_prompt import (
    ANSWER_FENCE_PROMPT_SCHEMA,
    answer_fence_prompt,
    strip_math_prompt_framing,
)
from postraining.opsd.data import TEACHER_PROMPT_SCHEMA, build_teacher_prompt
from postraining.opsd.manifest import (
    MATH_MIXTURE_DATA_SCHEMA,
    MATH_MIXTURE_GROUPS_PER_CYCLE,
    MATH_MIXTURE_SOURCE_QUOTAS,
    MATH_MIXTURE_SPLIT_SCHEMA,
)
from postraining.opsd.schemas import OPSD_PROMPT_SCHEMA
from postraining.prepare_sft_traces import INSTRUCTION_SUFFIX_ANSWER, word_ngrams


DEFAULT_DAPO_SOURCE = Path("postraining/data/dapo-math-17k.parquet")
DEFAULT_DEEPMIND_SOURCE = Path(
    "postraining/data/deepmind-interpolate-rl-full.parquet"
)
DEFAULT_GSM8K_SOURCE = Path("postraining/data/gsm8k_rl_prompts.parquet")
DEFAULT_SFT = Path(
    "postraining/data/sft_traces_v6_answer_bare_a1swap10k.parquet"
)
DEFAULT_OUTPUT_PREFIX = Path(
    "postraining/data/opsd_math_curriculum_24_18_6_sft6_bare"
)

DAPO_SOURCE_ID = "dapo_math_17k"
DEEPMIND_SOURCE_ID = "deepmind_math"
GSM8K_SOURCE_ID = "gsm8k"

MINERVA_REWARD_STYLE = "rule-lighteval/MATH_v2"
EXACT_REWARD_STYLE = "rule"
SOURCE_QUOTAS = MATH_MIXTURE_SOURCE_QUOTAS
GROUPS_PER_CYCLE = MATH_MIXTURE_GROUPS_PER_CYCLE


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    path_argument: str
    reward_style: str
    require_numeric: bool
    allow_missing_style: bool = False


SOURCE_SPECS = (
    SourceSpec(
        DEEPMIND_SOURCE_ID,
        "deepmind_source",
        EXACT_REWARD_STYLE,
        False,
    ),
    SourceSpec(
        GSM8K_SOURCE_ID,
        "gsm8k_source",
        MINERVA_REWARD_STYLE,
        True,
        allow_missing_style=True,
    ),
    SourceSpec(DAPO_SOURCE_ID, "dapo_source", MINERVA_REWARD_STYLE, True),
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(text: str) -> str:
    return " ".join(text.split()).lower()


def row_fingerprint(row: dict) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"))


def raw_example_id(row: dict, source_id: str) -> str:
    info = row.get("extra_info") or {}
    value = info.get("index")
    if value is None or not str(value).strip():
        raise ValueError(f"{source_id} row lacks extra_info.index")
    return str(value)


def deduplicate_source(path: str | Path, source_id: str) -> tuple[list[dict], int]:
    """Drop byte-equivalent curriculum repeats and reject ID conflicts."""
    unique: dict[str, dict] = {}
    fingerprints: dict[str, str] = {}
    physical_rows = 0
    parquet = pq.ParquetFile(path)
    required = {"prompt", "reward_model", "extra_info"}
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{path} lacks source columns {sorted(missing)}")
    for batch in parquet.iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            physical_rows += 1
            example_id = raw_example_id(row, source_id)
            fingerprint = row_fingerprint(row)
            previous = fingerprints.get(example_id)
            if previous is not None:
                if previous != fingerprint:
                    raise ValueError(
                        f"conflicting physical rows for {source_id} id "
                        f"{example_id}"
                    )
                continue
            fingerprints[example_id] = fingerprint
            unique[example_id] = row
    return [unique[key] for key in sorted(unique)], physical_rows


def _prompt_content(row: dict, source_id: str) -> str:
    messages = row.get("prompt")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{source_id} prompt must be a nonempty message list")
    contents = []
    for message in messages:
        if not isinstance(message, dict) or message.get("content") is None:
            raise ValueError(f"{source_id} prompt contains an invalid message")
        contents.append(str(message["content"]))
    return "".join(contents)


def canonical_source_record(row: dict, spec: SourceSpec) -> dict:
    """Adapt one verifier row to the current final-answer OPSD columns."""
    source_id = spec.source_id
    raw_id = raw_example_id(row, source_id)
    content = _prompt_content(row, source_id)
    problem, removed = strip_math_prompt_framing(content)
    if removed < 1 or not problem:
        raise ValueError(
            f"{source_id} prompt did not contain recognized removable framing"
        )

    reward_model = row.get("reward_model") or {}
    truth_value = reward_model.get("ground_truth")
    if truth_value is None or not str(truth_value).strip():
        raise ValueError(f"{source_id} row {raw_id} lacks a ground truth")
    truth = str(truth_value).strip()
    observed_style = reward_model.get("style")
    if observed_style is None and not spec.allow_missing_style:
        raise ValueError(f"{source_id} row {raw_id} lacks a reward style")
    if observed_style is not None and str(observed_style) != spec.reward_style:
        raise ValueError(
            f"unexpected {source_id} reward style {observed_style!r}; "
            f"expected {spec.reward_style!r}"
        )
    if spec.require_numeric and parse_numeric_answer(truth) is None:
        raise ValueError(f"non-numeric {source_id} truth {truth!r}")

    if source_id == DAPO_SOURCE_ID:
        if row.get("data_source") != "math_dapo" or row.get("ability") != "MATH":
            raise ValueError("unexpected DAPO source or ability")
    elif source_id == GSM8K_SOURCE_ID:
        if row.get("data_source") != "gsm8k_train" or row.get("ability") != "MATH":
            raise ValueError("unexpected GSM8K source or ability")

    return {
        "example_id": f"{source_id}:{raw_id}",
        "source": source_id,
        "problem": problem,
        "solution": truth,
        "reference_kind": "final_answer",
        "ground_truth": truth,
        "reward_style": spec.reward_style,
        "verified": True,
    }


def answer_key(record: dict) -> str:
    """Canonical equality key under the row's declared reward contract."""
    style = record["reward_style"]
    answer = str(record["solution"])
    # Cross-source fallback buckets can pair DeepMind's exact answers with
    # Minerva-graded numeric answers.  Treat every numerically equivalent
    # spelling as one conflict class even though exact grading is narrower:
    # this conservative key prevents (for example) a DeepMind ``5`` donor
    # from verifying a GSM8K ``5.0`` target under Minerva normalization.
    numeric = parse_numeric_answer(answer)
    if numeric is not None:
        return f"numeric:{numeric}"
    if style == EXACT_REWARD_STYLE:
        return answer.strip()
    if style == MINERVA_REWARD_STYLE:
        return "minerva:" + normalize_final_answer(answer)
    raise ValueError(f"unsupported reward style {style!r}")


def deduplicate_canonical_records(
    records: list[dict],
) -> tuple[list[dict], Counter[str], Counter[str]]:
    """Deduplicate equal truths and quarantine every conflicting problem."""
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_problem[normalized_text(str(record["problem"]))].append(record)

    unique: list[dict] = []
    duplicate_counts: Counter[str] = Counter()
    conflict_counts: Counter[str] = Counter()
    for group in by_problem.values():
        if len({answer_key(record) for record in group}) > 1:
            conflict_counts.update(str(record["source"]) for record in group)
            continue
        unique.append(group[0])
        duplicate_counts.update(str(record["source"]) for record in group[1:])
    return unique, duplicate_counts, conflict_counts


def split_rank(example_id: str, seed: int, *, purpose: str = "gate") -> bytes:
    return hashlib.sha256(
        f"{MATH_MIXTURE_SPLIT_SCHEMA}:{purpose}:{seed}:{example_id}".encode()
    ).digest()


def _can_derange(records: list[dict]) -> bool:
    counts = Counter(answer_key(record) for record in records)
    return bool(counts) and max(counts.values()) * 2 <= len(records)


def nonderangeable_bucket_keys(
    records: list[dict], bucket_key: Callable[[dict], Hashable]
) -> set[Hashable]:
    buckets: dict[Hashable, list[dict]] = defaultdict(list)
    for record in records:
        buckets[bucket_key(record)].append(record)
    return {key for key, bucket in buckets.items() if not _can_derange(bucket)}


def _derange_one_bucket(
    records: list[dict], seed: int, bucket: Hashable
) -> dict[str, dict]:
    if not _can_derange(records):
        raise ValueError(f"answer majority is too large in bucket {bucket!r}")
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[answer_key(record)].append(record)

    def rank(record: dict) -> bytes:
        return split_rank(record["example_id"], seed, purpose=f"donor:{bucket}")

    ordered = [
        record
        for value in sorted(groups)
        for record in sorted(groups[value], key=rank)
    ]
    shift = max(map(len, groups.values()))
    donors = ordered[shift:] + ordered[:shift]
    if any(answer_key(row) == answer_key(donor) for row, donor in zip(ordered, donors)):
        raise AssertionError("constructed answer derangement is not value-distinct")
    return dict(zip((row["example_id"] for row in ordered), donors, strict=True))


def deranged_answer_donors(records: list[dict], seed: int) -> dict[str, dict]:
    """Derange within answer position, preferring source/style-local donors."""
    position_buckets: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        position_buckets[int(record["answer_slot_tokens"])].append(record)

    donors: dict[str, dict] = {}
    for position in sorted(position_buckets):
        bucket = position_buckets[position]
        preferred: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for record in bucket:
            preferred[(record["source"], record["reward_style"])].append(record)
        if all(_can_derange(group) for group in preferred.values()):
            for key in sorted(preferred):
                donors.update(
                    _derange_one_bucket(
                        preferred[key], seed, (position, *key)
                    )
                )
        else:
            donors.update(_derange_one_bucket(bucket, seed, position))
    return donors


def sft_decontamination_index(
    path: str | Path,
) -> tuple[set[str], set[tuple[str, ...]]]:
    table = pq.read_table(path, columns=["problem"])
    exact: set[str] = set()
    ngrams: set[tuple[str, ...]] = set()
    for row in table.to_pylist():
        problem = str(row["problem"])
        exact.add(normalized_text(problem))
        ngrams |= word_ngrams(problem)
    return exact, ngrams


def output_paths(prefix: str | Path) -> tuple[Path, Path, Path]:
    prefix = Path(prefix)
    return (
        prefix.with_name(prefix.name + "_train.parquet"),
        prefix.with_name(prefix.name + "_gate.parquet"),
        prefix.with_name(prefix.name + ".manifest.json"),
    )


def require_fresh_outputs(paths: tuple[Path, ...]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite immutable OPSD math mixture artifacts: "
            + ", ".join(str(path) for path in existing)
        )


def _temporary_path(path: Path) -> Path:
    return path.with_name(path.name + f".{os.getpid()}.tmp")


def _publish_no_replace(temporary: Path, destination: Path) -> None:
    """Atomically publish one same-filesystem file only if absent."""
    os.link(temporary, destination)
    temporary.unlink()


def _write_outputs_atomically(
    train_rows: list[dict],
    gate_rows: list[dict],
    train_path: Path,
    gate_path: Path,
    manifest_path: Path,
    manifest_without_hashes: dict,
) -> dict:
    temporaries = {
        train_path: _temporary_path(train_path),
        gate_path: _temporary_path(gate_path),
        manifest_path: _temporary_path(manifest_path),
    }
    published: list[Path] = []
    try:
        pq.write_table(pa.Table.from_pylist(train_rows), temporaries[train_path])
        pq.write_table(pa.Table.from_pylist(gate_rows), temporaries[gate_path])
        manifest = {
            **manifest_without_hashes,
            "train_sha256": file_sha256(temporaries[train_path]),
            "gate_sha256": file_sha256(temporaries[gate_path]),
        }
        temporaries[manifest_path].write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        # The manifest is the commit marker and is installed last.
        _publish_no_replace(temporaries[train_path], train_path)
        published.append(train_path)
        _publish_no_replace(temporaries[gate_path], gate_path)
        published.append(gate_path)
        _publish_no_replace(temporaries[manifest_path], manifest_path)
        published.append(manifest_path)
        return manifest
    except BaseException:
        # A Python/OS error during publication must not strand an
        # uncommitted train or gate artifact.  The exclusive build lock
        # guarantees these paths were published by this invocation.
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)


def _source_gate_rows(gate_rows: int) -> dict[str, int]:
    if gate_rows < 1 or gate_rows % 8:
        raise ValueError("gate rows must be a positive multiple of 8 for 4:3:1")
    return {
        source: gate_rows * quota // GROUPS_PER_CYCLE
        for source, quota in SOURCE_QUOTAS.items()
    }


def _select_gate(
    candidates: list[dict], gate_rows: int, seed: int
) -> tuple[list[dict], set[int]]:
    target = _source_gate_rows(gate_rows)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for record in candidates:
        by_source[record["source"]].append(record)
    for source in by_source:
        by_source[source].sort(
            key=lambda record: split_rank(record["example_id"], seed)
        )

    excluded_positions: set[int] = set()
    while True:
        selected: list[dict] = []
        for source in SOURCE_QUOTAS:
            available = [
                record
                for record in by_source.get(source, [])
                if record["answer_slot_tokens"] not in excluded_positions
            ]
            if len(available) < target[source]:
                raise ValueError(
                    f"only {len(available)} clean {source} rows for gate quota "
                    f"{target[source]} after answer-position filtering"
                )
            selected.extend(available[: target[source]])
        bad = nonderangeable_bucket_keys(
            selected, lambda record: record["answer_slot_tokens"]
        )
        if not bad:
            return selected, excluded_positions
        excluded_positions.update(int(value) for value in bad)


def _attach_derangement(
    records: list[dict], tokenizer, seed: int
) -> list[dict]:
    donors = deranged_answer_donors(records, seed)
    split_ids = {record["example_id"] for record in records}
    enriched = []
    for record in records:
        donor = donors[record["example_id"]]
        if donor["example_id"] not in split_ids:
            raise AssertionError("answer donor escaped its data split")
        donor_truth = donor["solution"]
        permuted_teacher_tokens = len(
            encode_prompt(
                tokenizer,
                build_teacher_prompt(
                    record["problem"],
                    donor_truth,
                    "final_answer",
                    INSTRUCTION_SUFFIX_ANSWER,
                ),
            )
        )
        if permuted_teacher_tokens != record["teacher_prompt_tokens"]:
            raise AssertionError("correct and permuted teacher positions differ")
        grading_style = (
            "exact" if record["reward_style"] == EXACT_REWARD_STYLE else "minerva"
        )
        if verify_answer(
            "Answer: " + donor_truth,
            record["ground_truth"],
            grading_style,
            window=None,
        )[0]:
            raise AssertionError("permuted answer accidentally verifies correct")
        enriched.append(
            {
                **record,
                "permuted_solution": donor_truth,
                "permuted_donor_id": donor["example_id"],
                "permuted_teacher_prompt_tokens": permuted_teacher_tokens,
            }
        )
    if Counter(record["solution"] for record in enriched) != Counter(
        record["permuted_solution"] for record in enriched
    ):
        raise AssertionError("permuted answer multiset changed within split")
    if any(
        record["example_id"] == record["permuted_donor_id"] for record in enriched
    ):
        raise AssertionError("permuted answer self-donor")
    return enriched


def _json_args(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _acquire_build_lock(manifest_path: Path) -> tuple[int, Path]:
    lock_path = manifest_path.with_name(manifest_path.name + ".lock")
    for attempt in range(2):
        try:
            descriptor = os.open(
                lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
            )
            break
        except FileExistsError as error:
            try:
                owner = int(lock_path.read_text().removeprefix("pid=").strip())
                os.kill(owner, 0)
            except ProcessLookupError:
                lock_path.unlink(missing_ok=True)
                if attempt == 0:
                    continue
            except (OSError, ValueError):
                pass
            raise FileExistsError(
                f"another OPSD math mixture build holds {lock_path}"
            ) from error
    else:
        raise AssertionError("build-lock acquisition loop exhausted")
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
    except BaseException:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)
        raise
    return descriptor, lock_path


def build(args: argparse.Namespace) -> dict:
    paths = output_paths(args.output_prefix)
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    descriptor, lock_path = _acquire_build_lock(paths[2])
    try:
        return _build_reserved(args, paths)
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _build_reserved(
    args: argparse.Namespace, paths: tuple[Path, Path, Path]
) -> dict:
    train_path, gate_path, manifest_path = paths
    require_fresh_outputs((train_path, gate_path, manifest_path))

    tokenizer = GPT2BPETokenizer(think_tokens=True, answer_tokens=True)
    sft_path = Path(args.sft_corpus)
    sft_sha256 = file_sha256(sft_path)
    exact_sft, sft_ngrams = sft_decontamination_index(args.sft_corpus)
    rejections: Counter[str] = Counter()
    source_metadata: dict[str, dict] = {}
    eligible: list[dict] = []
    clean_gate_candidates: list[dict] = []
    all_canonical: list[dict] = []

    for spec in SOURCE_SPECS:
        path = Path(getattr(args, spec.path_argument))
        source_sha256 = file_sha256(path)
        source_rows, physical_rows = deduplicate_source(path, spec.source_id)
        canonical = [canonical_source_record(row, spec) for row in source_rows]
        if file_sha256(path) != source_sha256:
            raise ValueError(f"{spec.source_id} source changed while being read")
        all_canonical.extend(canonical)
        source_metadata[spec.source_id] = {
            "path": str(path),
            "sha256": source_sha256,
            "physical_rows": physical_rows,
            "unique_rows": len(canonical),
            "reward_style": spec.reward_style,
        }

    (
        canonical_records,
        normalized_duplicates,
        normalized_conflicts,
    ) = deduplicate_canonical_records(all_canonical)
    canonical_source_rows = Counter(
        record["source"] for record in canonical_records
    )
    for source, metadata in source_metadata.items():
        metadata["canonical_rows"] = canonical_source_rows[source]
        metadata["normalized_problem_duplicates"] = normalized_duplicates[source]
        metadata["normalized_problem_conflicts"] = normalized_conflicts[source]
        rejections[f"{source}/normalized_problem_conflict"] = (
            normalized_conflicts[source]
        )
        metadata["eligible_rows"] = 0
        metadata["clean_gate_candidates"] = 0

    for record in canonical_records:
        spec = next(spec for spec in SOURCE_SPECS if spec.source_id == record["source"])
        metadata = source_metadata[spec.source_id]
        student_tokens = len(
            encode_prompt(tokenizer, answer_fence_prompt(record["problem"]))
        )
        teacher_tokens = len(
            encode_prompt(
                tokenizer,
                build_teacher_prompt(
                    record["problem"],
                    record["solution"],
                    "final_answer",
                    INSTRUCTION_SUFFIX_ANSWER,
                ),
            )
        )
        answer_slot_tokens = len(
            encode_prompt(
                tokenizer,
                build_teacher_prompt(
                    "Length-control problem.",
                    record["solution"],
                    "final_answer",
                    INSTRUCTION_SUFFIX_ANSWER,
                ),
            )
        )
        if student_tokens > args.max_prompt_length:
            rejections[f"{spec.source_id}/student_prompt_overflow"] += 1
            continue
        if teacher_tokens + args.max_completion_length > args.context_tokens:
            rejections[f"{spec.source_id}/teacher_context_overflow"] += 1
            continue
        enriched = {
            **record,
            "student_prompt_tokens": student_tokens,
            "teacher_prompt_tokens": teacher_tokens,
            "answer_slot_tokens": answer_slot_tokens,
            "answer_fence_prompt_schema": ANSWER_FENCE_PROMPT_SCHEMA,
            "teacher_prompt_schema": TEACHER_PROMPT_SCHEMA,
            "opsd_prompt_schema": OPSD_PROMPT_SCHEMA,
        }
        eligible.append(enriched)
        metadata["eligible_rows"] += 1
        problem_ngrams = word_ngrams(record["problem"])
        if normalized_text(record["problem"]) in exact_sft:
            rejections[f"{spec.source_id}/gate_exact_sft_overlap"] += 1
        elif problem_ngrams & sft_ngrams:
            rejections[f"{spec.source_id}/gate_8gram_sft_overlap"] += 1
        else:
            clean_gate_candidates.append(enriched)
            metadata["clean_gate_candidates"] += 1

    if len({record["example_id"] for record in eligible}) != len(eligible):
        raise ValueError("canonical example IDs are not globally unique")

    gate_base, excluded_gate_positions = _select_gate(
        clean_gate_candidates, args.gate_rows, args.seed
    )
    rejections["gate_answer_position_bucket_excluded"] = sum(
        record["answer_slot_tokens"] in excluded_gate_positions
        for record in clean_gate_candidates
    )
    gate_ids = {record["example_id"] for record in gate_base}
    train_candidates = [
        record for record in eligible if record["example_id"] not in gate_ids
    ]
    bad_train_positions = nonderangeable_bucket_keys(
        train_candidates, lambda record: record["answer_slot_tokens"]
    )
    train_base = [
        record
        for record in train_candidates
        if record["answer_slot_tokens"] not in bad_train_positions
    ]
    rejections["train_answer_position_bucket_excluded"] = (
        len(train_candidates) - len(train_base)
    )
    if not train_base:
        raise ValueError("answer-position filtering left an empty train split")
    missing_train_sources = set(SOURCE_QUOTAS) - {
        record["source"] for record in train_base
    }
    if missing_train_sources:
        raise ValueError(
            "train split lacks quota sources "
            + ", ".join(sorted(missing_train_sources))
        )

    train_rows = _attach_derangement(train_base, tokenizer, args.seed + 1)
    gate_rows = _attach_derangement(gate_base, tokenizer, args.seed + 2)
    train_source_rows = dict(
        sorted(Counter(row["source"] for row in train_rows).items())
    )
    gate_source_rows = dict(
        sorted(Counter(row["source"] for row in gate_rows).items())
    )
    expected_gate_counts = _source_gate_rows(args.gate_rows)
    if gate_source_rows != dict(sorted(expected_gate_counts.items())):
        raise AssertionError("gate source composition differs from train quotas")
    if gate_ids & {record["example_id"] for record in train_rows}:
        raise AssertionError("held-out gate IDs leaked into train")

    for source, metadata in source_metadata.items():
        if file_sha256(metadata["path"]) != metadata["sha256"]:
            raise ValueError(f"{source} source changed during the build")
    if file_sha256(sft_path) != sft_sha256:
        raise ValueError("SFT corpus changed during the build")
    manifest_without_hashes = {
        "schema": MATH_MIXTURE_DATA_SCHEMA,
        "split_schema": MATH_MIXTURE_SPLIT_SCHEMA,
        "answer_fence_prompt_schema": ANSWER_FENCE_PROMPT_SCHEMA,
        "teacher_prompt_schema": TEACHER_PROMPT_SCHEMA,
        "opsd_prompt_schema": OPSD_PROMPT_SCHEMA,
        "groups_per_cycle": GROUPS_PER_CYCLE,
        "source_quotas": dict(SOURCE_QUOTAS),
        "sources": source_metadata,
        "source_hashes": {
            source: metadata["sha256"]
            for source, metadata in source_metadata.items()
        },
        "source_physical_rows": {
            source: metadata["physical_rows"]
            for source, metadata in source_metadata.items()
        },
        "source_unique_rows": {
            source: metadata["unique_rows"]
            for source, metadata in source_metadata.items()
        },
        "sft_corpus": str(sft_path),
        "sft_corpus_sha256": sft_sha256,
        "eligible_rows": len(eligible),
        "train": str(train_path),
        "train_rows": len(train_rows),
        "train_source_rows": train_source_rows,
        "train_ids_sha256": hashlib.sha256(
            "\n".join(sorted(record["example_id"] for record in train_rows)).encode()
        ).hexdigest(),
        "gate": str(gate_path),
        "gate_rows": len(gate_rows),
        "gate_source_rows": gate_source_rows,
        "gate_ids_sha256": hashlib.sha256(
            "\n".join(sorted(gate_ids)).encode()
        ).hexdigest(),
        "rejections": dict(sorted(rejections.items())),
        "args": _json_args(args),
    }
    return _write_outputs_atomically(
        train_rows,
        gate_rows,
        train_path,
        gate_path,
        manifest_path,
        manifest_without_hashes,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dapo-source", default=str(DEFAULT_DAPO_SOURCE))
    parser.add_argument("--deepmind-source", default=str(DEFAULT_DEEPMIND_SOURCE))
    parser.add_argument("--gsm8k-source", default=str(DEFAULT_GSM8K_SOURCE))
    parser.add_argument("--sft-corpus", default=str(DEFAULT_SFT))
    parser.add_argument("--output-prefix", default=str(DEFAULT_OUTPUT_PREFIX))
    # GSM8K is almost entirely present in SFT; after exact/8-gram exclusion
    # only enough clean rows remain for its 72-row share of this 192 panel.
    parser.add_argument("--gate-rows", type=int, default=192)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if min(
        args.gate_rows,
        args.max_prompt_length,
        args.max_completion_length,
        args.context_tokens,
    ) < 1:
        parser.error("row and token budgets must be positive")
    if args.max_completion_length >= args.context_tokens:
        parser.error("completion budget must be smaller than context")
    try:
        manifest = build(args)
    except (FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
