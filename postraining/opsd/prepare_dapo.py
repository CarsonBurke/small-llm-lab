"""Build immutable, deduplicated DAPO-17K data for OPSD and its uplift gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from postraining.core import (
    GPT2BPETokenizer,
    encode_prompt,
    parse_numeric_answer,
    verify_answer,
)
from postraining.math_prompt import (
    ANSWER_FENCE_PROMPT_SCHEMA,
    answer_fence_prompt,
    strip_math_prompt_framing,
)
from postraining.opsd.data import TEACHER_PROMPT_SCHEMA, build_teacher_prompt
from postraining.opsd.schemas import OPSD_PROMPT_SCHEMA
from postraining.prepare_sft_traces import INSTRUCTION_SUFFIX_ANSWER, word_ngrams


DAPO_OPSD_DATA_SCHEMA = "dapo_opsd_final_answer_privilege/v2"
DAPO_OPSD_SPLIT_SCHEMA = (
    "sha256_clean_gate_then_split_local_token_length_answer_derangement/v3"
)
DEFAULT_SOURCE = Path("postraining/data/dapo-math-17k.parquet")
DEFAULT_SFT = Path(
    "postraining/data/sft_traces_v4_answer_canonical_hfonly.parquet"
)
DEFAULT_OUTPUT_PREFIX = Path("postraining/data/opsd_dapo17k_contractlast")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(text: str) -> str:
    return " ".join(text.split()).lower()


def row_fingerprint(row: dict) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"))


def deduplicate_dapo(path: Path) -> tuple[list[dict], int]:
    """Deduplicate physical curriculum repeats and reject conflicting copies."""
    rows_by_id: dict[str, dict] = {}
    fingerprints: dict[str, str] = {}
    physical_rows = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            physical_rows += 1
            info = row.get("extra_info") or {}
            example_id = str(info.get("index") or "")
            if not example_id:
                raise ValueError("DAPO row lacks extra_info.index")
            fingerprint = row_fingerprint(row)
            previous = fingerprints.get(example_id)
            if previous is not None:
                if previous != fingerprint:
                    raise ValueError(
                        f"conflicting physical rows for DAPO id {example_id}"
                    )
                continue
            fingerprints[example_id] = fingerprint
            rows_by_id[example_id] = row
    return [rows_by_id[key] for key in sorted(rows_by_id)], physical_rows


def canonical_dapo_record(row: dict) -> dict:
    if row.get("data_source") != "math_dapo" or row.get("ability") != "MATH":
        raise ValueError("unexpected DAPO source or ability")
    messages = row.get("prompt")
    if not isinstance(messages, list) or not messages:
        raise ValueError("DAPO prompt must be a nonempty message list")
    content = "".join(str(message.get("content") or "") for message in messages)
    problem, removed = strip_math_prompt_framing(content)
    if removed < 1 or not problem:
        raise ValueError("DAPO prompt did not contain recognized removable framing")
    reward_model = row.get("reward_model") or {}
    truth = str(reward_model.get("ground_truth") or "").strip()
    style = str(reward_model.get("style") or "")
    if style != "rule-lighteval/MATH_v2":
        raise ValueError(f"unexpected DAPO reward style {style!r}")
    if parse_numeric_answer(truth) is None:
        raise ValueError(f"non-numeric DAPO truth {truth!r}")
    example_id = str((row.get("extra_info") or {}).get("index"))
    return {
        "example_id": example_id,
        "source": "dapo_math_17k",
        "problem": problem,
        "solution": truth,
        "reference_kind": "final_answer",
        "ground_truth": truth,
        "reward_style": style,
        "verified": True,
    }


def deranged_answer_donors(
    records: list[dict],
    seed: int,
    *,
    bucket_key=None,
) -> dict[str, dict]:
    """Return a deterministic multiset-preserving wrong-answer assignment."""
    if bucket_key is None:
        bucket_key = lambda record: None
    buckets: dict[object, list[dict]] = defaultdict(list)
    for record in records:
        buckets[bucket_key(record)].append(record)
    donors_by_id = {}
    for bucket in sorted(buckets, key=str):
        donors_by_id.update(
            _deranged_answer_donors_one_bucket(buckets[bucket], seed, bucket)
        )
    return donors_by_id


def _deranged_answer_donors_one_bucket(
    records: list[dict], seed: int, bucket: object
) -> dict[str, dict]:
    groups: dict[object, list[dict]] = defaultdict(list)
    for record in records:
        value = parse_numeric_answer(record["solution"])
        if value is None:
            raise ValueError("derangement requires numeric answers")
        groups[value].append(record)
    largest = max(map(len, groups.values()), default=0)
    if largest * 2 > len(records):
        raise ValueError("answer majority is too large for a valid derangement")

    def rank(record: dict) -> bytes:
        return hashlib.sha256(
            f"{DAPO_OPSD_DATA_SCHEMA}:{seed}:{bucket}:{record['example_id']}".encode()
        ).digest()

    ordered = [
        record
        for value in sorted(groups, key=str)
        for record in sorted(groups[value], key=rank)
    ]
    for shift in range(largest, len(ordered)):
        donors = ordered[shift:] + ordered[:shift]
        if all(
            parse_numeric_answer(record["solution"])
            != parse_numeric_answer(donor["solution"])
            for record, donor in zip(ordered, donors, strict=True)
        ):
            return {
                record["example_id"]: donor
                for record, donor in zip(ordered, donors, strict=True)
            }
    raise ValueError("could not construct a value-distinct answer derangement")


def nonderangeable_bucket_keys(records: list[dict], bucket_key) -> set[object]:
    """Buckets whose answer-value majority makes derangement impossible."""
    buckets: dict[object, Counter] = defaultdict(Counter)
    for record in records:
        value = parse_numeric_answer(record["solution"])
        buckets[bucket_key(record)][value] += 1
    return {
        bucket
        for bucket, counts in buckets.items()
        if not counts or max(counts.values()) * 2 > sum(counts.values())
    }


def sft_decontamination_index(path: Path) -> tuple[set[str], set[tuple[str, ...]]]:
    table = pq.read_table(path, columns=["problem"])
    exact: set[str] = set()
    ngrams: set[tuple[str, ...]] = set()
    for row in table.to_pylist():
        problem = str(row["problem"])
        exact.add(normalized_text(problem))
        ngrams |= word_ngrams(problem)
    return exact, ngrams


def split_rank(example_id: str, seed: int) -> bytes:
    return hashlib.sha256(
        f"{DAPO_OPSD_SPLIT_SCHEMA}:{seed}:{example_id}".encode()
    ).digest()


def output_paths(prefix: Path) -> tuple[Path, Path, Path]:
    return (
        prefix.with_name(prefix.name + "_train.parquet"),
        prefix.with_name(prefix.name + "_gate.parquet"),
        prefix.with_name(prefix.name + ".manifest.json"),
    )


def require_fresh_outputs(paths: tuple[Path, ...]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing DAPO OPSD artifacts: "
            + ", ".join(str(path) for path in existing)
        )


def atomic_write_parquet(rows: list[dict], path: Path) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    os.replace(temporary, path)


def atomic_write_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build(args: argparse.Namespace) -> dict:
    source = Path(args.source)
    sft_corpus = Path(args.sft_corpus)
    train_path, gate_path, manifest_path = output_paths(Path(args.output_prefix))
    require_fresh_outputs((train_path, gate_path, manifest_path))
    train_path.parent.mkdir(parents=True, exist_ok=True)

    physical_rows_data, physical_rows = deduplicate_dapo(source)
    records = [canonical_dapo_record(row) for row in physical_rows_data]
    exact, sft_ngrams = sft_decontamination_index(sft_corpus)
    tokenizer = GPT2BPETokenizer(think_tokens=True, answer_tokens=True)
    rejection_counts: Counter[str] = Counter()
    eligible_base: list[dict] = []
    clean_ids: set[str] = set()
    for record in records:
        student_prompt = answer_fence_prompt(record["problem"])
        teacher_prompt = build_teacher_prompt(
            record["problem"],
            record["solution"],
            "final_answer",
            INSTRUCTION_SUFFIX_ANSWER,
        )
        student_tokens = len(encode_prompt(tokenizer, student_prompt))
        teacher_tokens = len(encode_prompt(tokenizer, teacher_prompt))
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
            rejection_counts["student_prompt_overflow"] += 1
            continue
        if teacher_tokens + args.max_completion_length > args.context_tokens:
            rejection_counts["teacher_context_overflow"] += 1
            continue
        eligible_base.append(
            {
                **record,
                "student_prompt_tokens": student_tokens,
                "teacher_prompt_tokens": teacher_tokens,
                "answer_slot_tokens": answer_slot_tokens,
                "answer_fence_prompt_schema": ANSWER_FENCE_PROMPT_SCHEMA,
                "teacher_prompt_schema": TEACHER_PROMPT_SCHEMA,
                "opsd_prompt_schema": OPSD_PROMPT_SCHEMA,
            }
        )
        problem_ngrams = word_ngrams(record["problem"])
        exact_overlap = normalized_text(record["problem"]) in exact
        ngram_overlap = bool(problem_ngrams & sft_ngrams)
        if exact_overlap:
            rejection_counts["gate_exact_sft_overlap"] += 1
        elif ngram_overlap:
            rejection_counts["gate_8gram_sft_overlap"] += 1
        else:
            clean_ids.add(record["example_id"])

    clean_gate_candidates = [
        record for record in eligible_base if record["example_id"] in clean_ids
    ]

    clean_gate_candidates.sort(
        key=lambda record: split_rank(record["example_id"], args.seed)
    )
    if len(clean_gate_candidates) < args.gate_rows:
        raise ValueError(
            f"only {len(clean_gate_candidates)} clean rows for "
            f"--gate-rows {args.gate_rows}"
        )
    globally_bad_gate_buckets = nonderangeable_bucket_keys(
        clean_gate_candidates,
        lambda record: record["answer_slot_tokens"],
    )
    excluded_gate_buckets = set(globally_bad_gate_buckets)
    while True:
        gate_base = [
            record
            for record in clean_gate_candidates
            if record["answer_slot_tokens"] not in excluded_gate_buckets
        ][: args.gate_rows]
        if len(gate_base) < args.gate_rows:
            raise ValueError(
                "not enough clean rows after exact-position control filtering"
            )
        bad = nonderangeable_bucket_keys(
            gate_base, lambda record: record["answer_slot_tokens"]
        )
        if not bad:
            break
        excluded_gate_buckets.update(bad)
    rejection_counts["gate_answer_position_bucket_excluded"] += sum(
        record["answer_slot_tokens"] in excluded_gate_buckets
        for record in clean_gate_candidates
    )
    gate_ids = {record["example_id"] for record in gate_base}
    train_candidates = [
        record for record in eligible_base if record["example_id"] not in gate_ids
    ]
    bad_train_buckets = nonderangeable_bucket_keys(
        train_candidates, lambda record: record["answer_slot_tokens"]
    )
    train_base = [
        record
        for record in train_candidates
        if record["answer_slot_tokens"] not in bad_train_buckets
    ]
    rejection_counts["train_answer_position_bucket_excluded"] += (
        len(train_candidates) - len(train_base)
    )

    def attach_split_derangement(split: list[dict], split_seed: int) -> list[dict]:
        donors = deranged_answer_donors(
            split,
            split_seed,
            bucket_key=lambda record: record["answer_slot_tokens"],
        )
        enriched = []
        split_ids = {record["example_id"] for record in split}
        for record in split:
            donor = donors[record["example_id"]]
            donor_truth = donor["solution"]
            if donor["example_id"] not in split_ids:
                raise AssertionError("answer donor escaped its data split")
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
                raise AssertionError(
                    "correct and permuted teacher response positions differ"
                )
            if verify_answer(
                "Answer: " + donor_truth,
                record["ground_truth"],
                "minerva",
                window=None,
            )[0]:
                raise AssertionError(
                    "permuted answer accidentally verifies correct"
                )
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
            record["example_id"] == record["permuted_donor_id"]
            for record in enriched
        ):
            raise AssertionError("permuted answer self-donor")
        return enriched

    train_rows = attach_split_derangement(train_base, args.seed + 1)
    gate_rows = attach_split_derangement(gate_base, args.seed + 2)
    atomic_write_parquet(train_rows, train_path)
    atomic_write_parquet(gate_rows, gate_path)

    manifest = {
        "schema": DAPO_OPSD_DATA_SCHEMA,
        "split_schema": DAPO_OPSD_SPLIT_SCHEMA,
        "answer_fence_prompt_schema": ANSWER_FENCE_PROMPT_SCHEMA,
        "teacher_prompt_schema": TEACHER_PROMPT_SCHEMA,
        "opsd_prompt_schema": OPSD_PROMPT_SCHEMA,
        "source": str(source),
        "source_sha256": file_sha256(source),
        "sft_corpus": str(sft_corpus),
        "sft_corpus_sha256": file_sha256(sft_corpus),
        "physical_rows": physical_rows,
        "unique_rows": len(records),
        "eligible_rows": len(eligible_base),
        "train_rows": len(train_rows),
        "gate_rows": len(gate_rows),
        "clean_gate_candidates": len(clean_gate_candidates),
        "rejections": dict(sorted(rejection_counts.items())),
        "train": str(train_path),
        "train_sha256": file_sha256(train_path),
        "gate": str(gate_path),
        "gate_sha256": file_sha256(gate_path),
        "gate_ids_sha256": hashlib.sha256(
            "\n".join(sorted(gate_ids)).encode()
        ).hexdigest(),
        "args": vars(args),
    }
    atomic_write_json(manifest, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--sft-corpus", default=str(DEFAULT_SFT))
    parser.add_argument("--output-prefix", default=str(DEFAULT_OUTPUT_PREFIX))
    parser.add_argument("--gate-rows", type=int, default=256)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
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
