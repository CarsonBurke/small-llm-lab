#!/usr/bin/env python3
"""Build an immutable, conservatively decontaminated OpenMathReasoning archive.

This is CPU data preparation, not a model run. Size means UTF-8 bytes of
problem + generated_solution, not compressed download size or token count.
No trace is truncated. Passing these screens does not prove answer correctness
or the absence of semantic paraphrases.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from postraining.math_prompt import strip_math_prompt_framing
from postraining.choice_prompt import split_options
from postraining.problem_overlap import ProblemOverlapIndex, template_shingles, whitespace_key

DEFAULT_RL = ROOT / "postraining/data/reviewed_candidate_20260926_v2/mixture.manifest.json"
EVALUATION_PATHS = (
    "postraining/data/relaxed_bar/gsm8k_test_questions.parquet",
    "postraining/data/deepmind-interpolate-easy.parquet",
    "postraining/data/aime-2024.parquet",
    "postraining/data/aime-2025.parquet",
    "postraining/data/aime-2026.parquet",
    "postraining/data/relaxed_bar/gsm8k_main_train.parquet",
    "postraining/data/relaxed_bar/gsm8k_socratic_train.parquet",
    "postraining/data/gsm8k_rl_prompts.parquet",
    "postraining/data/readiness_20260926_v1/science_mc.parquet",
)
RL_NAMES = {"deepmind_easy", "ultradata_math", "dapo", "ultradata_code_l3",
            "ultradata_knowledge", "science_mc"}
RESERVED = re.compile(r"</?(?:think|answer|tool_call|tool_response)>|<\|[^\n<>]*\|>|<tool_call|<function_call", re.I)
BOX = re.compile(r"\\boxed\s*\{")
HISTOGRAM_EDGES = (1024, 4096, 8192, 16384, 32768, 65536, 131072)
ALLOWED_SOURCES = {"aops_c6_high_school_olympiads", "aops_c4_high_school_math", "aops_c7_college_math",
                   "aops_c5_contests_amp_programs", "MATH_training_set"}
ALLOWED_PROBLEM_TYPES = {"has_answer_extracted", "no_answer_extracted", "converted_proof"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def canonical_problem(row: dict) -> str:
    extra = row.get("extra_info") or {}
    raw = extra.get("raw_problem") if isinstance(extra, dict) else None
    if not raw:
        raw = row.get("problem") or row.get("question")
    if not raw:
        prompts = row.get("prompt")
        users = [p["content"] for p in prompts or [] if p.get("role") == "user"]
        if len(users) != 1:
            raise ValueError("Reference row must have exactly one user question")
        raw = users[0]
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Empty or malformed reference question")
    return strip_math_prompt_framing(raw)[0]


def terminal_box(solution: str) -> str | None:
    """Extract a balanced last box only if nothing substantive follows it."""
    matches = list(BOX.finditer(solution))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for position in range(start, len(solution)):
        char = solution[position]
        if char in "{}" and (position == 0 or solution[position - 1] != "\\"):
            depth += 1 if char == "{" else -1
        if depth == 0:
            tail = solution[position + 1:].strip()
            if re.sub(r"\s|\$|\\[\])]|[.*]", "", tail):
                return None
            answer = solution[start:position].strip()
            return answer or None
    return None


def answer_key(answer: str) -> str:
    answer = answer.strip().strip("$")
    # Upstream answers sometimes escape commands twice. In environments,
    # however, two backslashes are meaningful row separators even immediately
    # before a letter. Do not guess at escaping anywhere in such an answer.
    if not re.search(r"\\(?:begin|end)(?![A-Za-z])", answer):
        answer = re.sub(r"\\\\(?=[A-Za-z(\[)\]])", r"\\", answer)
    if (answer.startswith(r"\(") and answer.endswith(r"\)")) or (answer.startswith(r"\[") and answer.endswith(r"\]")):
        answer = answer[2:-2]
    # Match complete control words, not prefixes (e.g. leftarrow/rightarrow),
    # and never consume the second backslash of an untouched row separator.
    answer = re.sub(r"(?<!\\)\\[dt]frac(?![A-Za-z])", r"\\frac", answer)
    answer = re.sub(r"(?<!\\)\\(?:left|right)(?![A-Za-z])", "", answer)
    answer = re.sub(r"(?<!\\)\\[,!]", "", answer)
    return re.sub(r"\s+", "", answer)


def normalize_row(row: dict) -> dict:
    """Remove only valid teacher thinking wrappers; preserve the exact source."""
    solution = row.get("generated_solution")
    if not isinstance(solution, str):
        return row
    if solution.startswith("<think>") and solution.count("<think>") == 1 and solution.count("</think>") == 1:
        solution = solution[len("<think>"):].replace("</think>", "", 1)
    problem = row.get("problem")
    normalized = {**row, "source_solution_sha256": hashlib.sha256(row["generated_solution"].encode()).hexdigest(),
                  "generated_solution": solution}
    if isinstance(problem, str):
        normalized.update(problem=strip_math_prompt_framing(problem)[0],
                          source_problem_sha256=hashlib.sha256(problem.encode()).hexdigest())
    return normalized


def reference_questions(rows: list[dict]) -> list[str]:
    questions = []
    for row in rows:
        question = canonical_problem(row)
        questions.append(question)
        choices = split_options(question)
        if choices:
            questions.append(choices.question)
    return list(dict.fromkeys(questions))


def stem_targets() -> tuple[tuple[Path, str], ...]:
    """Evaluate the repository's literal path declaration without importing its trainer dependencies."""
    module = ast.parse((ROOT / "postraining/prepare_sft_corpus.py").read_text())
    declaration = next(node.value for node in module.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "STEM_QUESTION_TARGETS" for target in node.targets))
    return eval(compile(ast.Expression(declaration), "STEM_QUESTION_TARGETS", "eval"), {"Path": Path, "__builtins__": {}})


def quality_reason(row: dict) -> str | None:
    problem, solution = row.get("problem"), row.get("generated_solution")
    if not isinstance(problem, str) or not isinstance(solution, str) or not problem.strip() or not solution.strip():
        return "empty_or_malformed_text"
    if row.get("inference_mode") != "cot":
        return "non_cot"
    if row.get("problem_source") not in ALLOWED_SOURCES:
        return "unknown_problem_source"
    if row.get("problem_type") not in ALLOWED_PROBLEM_TYPES:
        return "unknown_answer_origin"
    if "\x00" in problem + solution or "\ufffd" in problem + solution:
        return "invalid_text"
    if RESERVED.search(problem + solution):
        return "reserved_fence_or_tool_artifact"
    answer = terminal_box(solution)
    if answer is None:
        return "no_terminal_boxed_answer"
    expected = row.get("expected_answer")
    if not isinstance(expected, str) or not expected.strip():
        return "missing_expected_answer"
    if RESERVED.search(expected):
        return "reserved_answer_token"
    if answer_key(answer) != answer_key(expected):
        return "boxed_expected_text_disagreement"
    if "final_answer" in row and row["final_answer"] != answer:
        return "final_answer_mismatch"
    return None


def load_references(rl_manifest: Path, evaluation_paths: tuple[Path, ...]) -> tuple[ProblemOverlapIndex, dict]:
    manifest = json.loads(rl_manifest.read_text())
    sources = manifest["sources"]
    if len(sources) != 6 or {s["name"] for s in sources} != RL_NAMES:
        raise ValueError("Exactly the six reviewed RL pools are required")
    pools, records = [], []
    for source in sources:
        path = Path(source["path"])
        if not path.is_absolute():
            path = ROOT / path
        digest = sha256(path)
        if digest != source["sha256"]:
            raise ValueError(f"RL hash mismatch: {path}")
        rows = pq.read_table(path).to_pylist()
        if len(rows) != source["rows"]:
            raise ValueError(f"RL row-count mismatch: {path}")
        pools.append(reference_questions(rows))
        records.append({"kind": "rl", "name": source["name"], "path": str(path), "sha256": digest,
                        "rows": len(rows), "unique_questions": len(pools[-1])})
    for path in evaluation_paths:
        digest = sha256(path)
        rows = pq.read_table(path).to_pylist()
        if not rows:
            raise ValueError(f"Empty protected evaluation: {path}")
        pools.append(reference_questions(rows))
        records.append({"kind": "evaluation_or_reserved", "path": str(path), "sha256": digest,
                        "rows": len(rows), "unique_questions": len(pools[-1])})
    for relative, column in stem_targets():
        path = ROOT / relative
        digest = sha256(path)
        if path.suffix == ".csv":
            with path.open(newline="") as handle:
                texts = [row[column] for row in csv.DictReader(handle)]
        else:
            texts = pq.read_table(path, columns=[column]).column(column).to_pylist()
        # Missing pre-revision GPQA questions are explicitly absent, not malformed rows.
        texts = [text for text in texts if isinstance(text, str) and text.strip()]
        if not texts:
            raise ValueError(f"Empty STEM reference column: {path}:{column}")
        pools.append(list(dict.fromkeys(texts)))
        records.append({"kind": "stem_evaluation", "path": str(path), "column": column,
                        "sha256": digest, "rows": len(texts), "unique_questions": len(pools[-1])})
    questions = [question for pool in pools for question in pool]
    index = ProblemOverlapIndex(questions, template=template_shingles(pools))
    # Positive controls cover every indexed identity, plus exercise the public
    # matching path at both ends of each source pool. Avoid quadratic template
    # work from calling the full shingle matcher on every exact self-match.
    for position, question in enumerate(questions):
        if position not in index._whitespace.get(whitespace_key(question), ()):
            raise ValueError("Protected question missing from overlap index")
    for pool in pools:
        if not index.matches(pool[0]) or not index.matches(pool[-1]):
            raise ValueError("Protected-question positive control failed")
    return index, {"rl_manifest": str(rl_manifest.resolve()), "rl_manifest_sha256": sha256(rl_manifest),
                   "references": records, "matcher": index.provenance(),
                   "positive_controls": {"exact_index_identities": len(questions), "public_match_queries": 2 * len(pools)}}


def verify_input(input_dir: Path, item: dict, deadline: float | None = None) -> dict:
    path = input_dir / item["path"]
    while not path.is_file() and deadline is not None and time.monotonic() < deadline:
        print(json.dumps({"waiting_for_input": item["path"]}), flush=True)
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
        raise ValueError(f"Input hash/size mismatch: {path}")
    file = pq.ParquetFile(path)
    required = {"problem", "generated_solution", "expected_answer", "inference_mode"}
    if not required <= set(file.schema_arrow.names):
        raise ValueError(f"Missing source columns: {path}")
    return {**item, "rows": file.metadata.num_rows}


def validate_inputs(input_dir: Path, allow_pending: bool = False) -> tuple[list[dict], dict]:
    manifest_path = input_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("repo_id") != "nvidia/OpenMathReasoning" or not manifest.get("revision"):
        raise ValueError("Expected a pinned nvidia/OpenMathReasoning source manifest")
    files = sorted(manifest["files"], key=lambda item: item["path"])
    if not files or len({item["path"] for item in files}) != len(files):
        raise ValueError("Missing or duplicate input shards")
    declared = {item["path"] for item in files}
    actual = {str(p.relative_to(input_dir)) for p in input_dir.rglob("*.parquet")}
    if (actual - declared) or (not allow_pending and actual != declared):
        raise ValueError(f"Input shard set mismatch: missing={len(declared-actual)}, extra={len(actual-declared)}")
    verified = []
    for item in files:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Source path escapes input directory")
        if not isinstance(item.get("bytes"), int) or item["bytes"] <= 0 or not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")):
            raise ValueError("Invalid pinned shard metadata")
        verified.append(item if allow_pending else verify_input(input_dir, item))
    return verified, {**manifest, "manifest_sha256": sha256(manifest_path)}


def length_bucket(length: int) -> str:
    return next((f"<{edge}" for edge in HISTOGRAM_EDGES if length < edge), f">={HISTOGRAM_EDGES[-1]}")


class PreparationPaused(Exception):
    """The current output batch is durable; expensive auditing may be retried."""


def audit_output(directory: Path, index: ProblemOverlapIndex, expected: list[dict], *, should_stop=None) -> dict:
    """Read back every retained row; do not trust the builder's match cache."""
    declared = {record["path"] for record in expected}
    if len(declared) != len(expected) or declared != {path.name for path in directory.glob("*.parquet")}:
        raise ValueError("Output shard set differs from manifest")
    seen, pairs = set(), set()
    rows = raw_bytes = 0
    for record in expected:
        path = directory / record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Output hash mismatch: {path}")
        shard_rows = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            if should_stop is not None and should_stop():
                raise PreparationPaused
            for row in batch.to_pylist():
                if quality_reason(row):
                    raise ValueError(f"Output quality audit failed: {path}")
                problem, solution = row["problem"], row["generated_solution"]
                pair = hashlib.sha256(problem.encode() + b"\x00" + solution.encode()).digest()
                if pair in pairs:
                    raise ValueError("Duplicate question/solution in retained output")
                pairs.add(pair)
                if problem not in seen:
                    if index.matches(problem):
                        raise ValueError("Retained question overlaps protected reference")
                    seen.add(problem)
                rows += 1
                shard_rows += 1
                raw_bytes += len(problem.encode()) + len(solution.encode())
        if shard_rows != record["rows"]:
            raise ValueError(f"Output row-count mismatch: {path}")
    return {"schema": "disjoint_math_sft_audit/v1", "rows": rows, "unique_questions": len(seen),
            "raw_text_bytes": raw_bytes, "question_solution_duplicates": 0, "protected_question_matches": 0,
            "method": "independent full retained-row parquet rescan; fresh unique-question match cache",
            "limitations": "Question-level screening only; unrelated protected questions quoted solely inside solutions are not screened. No semantic paraphrase proof or mathematical correctness verification; no token/context-fit claim."}


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _checkpoint_digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _read_checkpoint(staging: Path) -> dict:
    path = staging / "checkpoint.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("Resume requires a regular committed checkpoint.json")
    envelope = json.loads(path.read_text())
    if envelope.get("schema") != "disjoint_math_sft_checkpoint/v1" or envelope.get("payload_sha256") != _checkpoint_digest(envelope["payload"]):
        raise ValueError("Checkpoint checksum/schema mismatch")
    return envelope["payload"]


def _checkpoint_bindings(input_dir: Path, output_dir: Path, rl_manifest: Path,
                         evaluation_paths: tuple[Path, ...], source: dict, protection: dict,
                         target_bytes: int, shard_bytes: int) -> dict:
    dependencies = [Path(__file__), *(ROOT / "postraining" / name for name in (
        "problem_overlap.py", "math_prompt.py", "choice_prompt.py", "decontaminate.py", "problem_registry.py", "core.py"))]
    return {"input_dir": str(input_dir.resolve()), "output_dir": str(output_dir.resolve()),
            "rl_manifest": str(rl_manifest.resolve()), "evaluation_paths": [str(path.resolve()) for path in evaluation_paths],
            "source": source, "protection": protection, "target_bytes": target_bytes, "shard_bytes": shard_bytes,
            "code": {str(path.resolve()): sha256(path) for path in dependencies},
            "python": list(sys.version_info[:3]), "pyarrow": pa.__version__}


def _resume_files(staging: Path, outputs: list[dict], data_complete: bool) -> list[Path]:
    """Identify only this writer's one possible uncommitted output and temps."""
    declared = {item["path"] for item in outputs}
    if [item["path"] for item in outputs] != [f"part-{number:05d}.parquet" for number in range(len(outputs))]:
        raise ValueError("Checkpoint output names are not the committed contiguous prefix")
    next_name = f"part-{len(outputs):05d}.parquet"
    allowed_orphans = {next_name, next_name + ".tmp", "checkpoint.json.tmp"}
    if data_complete:
        allowed_orphans.update({name + suffix for name in ("audit.json", "manifest.json", "COMPLETE.json") for suffix in ("", ".tmp")})
    orphans = []
    for path in staging.iterdir():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Unknown non-regular staging entry: {path.name}")
        if path.name in declared or path.name == "checkpoint.json":
            continue
        if path.name not in allowed_orphans:
            raise ValueError(f"Unknown staging file: {path.name}")
        orphans.append(path)
    return orphans


def build(input_dir: Path, output_dir: Path, rl_manifest: Path, target_bytes: int,
          evaluation_paths: tuple[Path, ...], shard_bytes: int = 512_000_000,
          wait_for_input_seconds: float = 0, *, resume: bool = False,
          stop_after_input_shards: int | None = None, should_stop=None) -> dict:
    if target_bytes <= 0 or shard_bytes <= 0 or wait_for_input_seconds < 0:
        raise ValueError("Byte budgets must be positive and wait duration nonnegative")
    if stop_after_input_shards is not None and stop_after_input_shards <= 0:
        raise ValueError("stop_after_input_shards must be positive")
    should_stop = should_stop or (lambda: False)
    staging = output_dir.with_name(output_dir.name + ".building")
    if output_dir.exists() or (staging.exists() and not resume):
        raise FileExistsError("Output or staging exists; use a fresh destination or explicit --resume")
    if resume and (not staging.is_dir() or staging.is_symlink()):
        raise ValueError("Resume requires the existing regular .building directory")
    saved = _read_checkpoint(staging) if resume else None
    deadline = time.monotonic() + wait_for_input_seconds if wait_for_input_seconds > 0 else None
    inputs, source_manifest = validate_inputs(input_dir, allow_pending=deadline is not None)
    index, protection = load_references(rl_manifest, evaluation_paths)
    binding = _checkpoint_bindings(input_dir, output_dir, rl_manifest, evaluation_paths,
                                   source_manifest, protection, target_bytes, shard_bytes)
    if saved is not None and saved["binding"] != binding:
        raise ValueError("Checkpoint source, policy, code, or configuration binding mismatch")
    if deadline is not None:
        inputs[0] = verify_input(input_dir, inputs[0], deadline)
    input_schema = pq.ParquetFile(input_dir / inputs[0]["path"]).schema_arrow
    if {"source_solution_sha256", "source_problem_sha256", "source_shard", "source_row", "final_answer"} & set(input_schema.names):
        raise ValueError("Input already contains normalized-source column")
    schema = (input_schema.append(pa.field("source_solution_sha256", pa.string()))
              .append(pa.field("source_problem_sha256", pa.string()))
              .append(pa.field("source_shard", pa.string())).append(pa.field("source_row", pa.int64()))
              .append(pa.field("final_answer", pa.string())).remove_metadata())
    counts = Counter({"examined": 0, "retained": 0})
    histogram, sources, models, answer_origins = (Counter() for _ in range(4))
    problem_matches, pairs, repetitions = {}, set(), Counter()
    outputs, pending, consumed_inputs = [], [], []
    raw_bytes = pending_bytes = 0
    next_input = next_row = 0
    data_complete = False

    def accumulate(row: dict) -> None:
        nonlocal raw_bytes
        problem, solution = row["problem"], row["generated_solution"]
        pair = hashlib.sha256(problem.encode() + b"\x00" + solution.encode()).digest()
        if pair in pairs:
            raise ValueError("Duplicate committed question/solution")
        pairs.add(pair)
        raw_bytes += len(problem.encode()) + len(solution.encode())
        repetitions[problem] += 1
        histogram[length_bucket(len(problem) + len(solution))] += 1
        sources[str(row.get("problem_source"))] += 1
        models[str(row.get("generation_model"))] += 1
        answer_origins[str(row.get("problem_type", "unspecified_upstream"))] += 1

    if saved is not None:
        next_input, next_row = saved["next_input"], saved["next_row"]
        if not isinstance(next_input, int) or not isinstance(next_row, int) or not 0 <= next_input <= len(inputs) or next_row < 0:
            raise ValueError("Invalid checkpoint cursor")
        outputs = saved["outputs"]
        consumed_inputs = saved["verified_inputs"]
        if len(consumed_inputs) not in {next_input, min(next_input + 1, len(inputs))}:
            raise ValueError("Checkpoint verified-input prefix disagrees with cursor")
        for number, item in enumerate(consumed_inputs):
            current = inputs[number]
            if "rows" not in current:
                current = verify_input(input_dir, current, deadline)
                inputs[number] = current
            if item != current:
                raise ValueError("Checkpoint verified input changed")
        if next_row and (next_input >= len(consumed_inputs) or next_row > consumed_inputs[next_input]["rows"]):
            raise ValueError("Checkpoint row cursor exceeds verified input")
        if next_input == len(inputs) and next_row:
            raise ValueError("Finished-input checkpoint has nonzero row cursor")
        counts = Counter(saved["counts"])
        if any(not isinstance(value, int) or value < 0 for value in counts.values()):
            raise ValueError("Invalid checkpoint counters")
        examined = sum(item["rows"] for item in consumed_inputs[:next_input]) + next_row
        if counts["examined"] != examined or sum(value for key, value in counts.items() if key != "examined") != examined:
            raise ValueError("Checkpoint counts disagree with processed-input cursor")
        data_complete = saved["data_complete"]
        if not isinstance(data_complete, bool) or data_complete != (saved["raw_text_bytes"] >= target_bytes or next_input == len(inputs)):
            raise ValueError("Checkpoint completion flag disagrees with cursor/byte budget")
        orphans = _resume_files(staging, outputs, data_complete)
        restored_rows = 0
        for item in outputs:
            path = staging / item["path"]
            if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                raise ValueError("Committed output hash/size mismatch")
            shard_rows = shard_bytes_read = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    if quality_reason(row):
                        raise ValueError("Committed output quality check failed")
                    accumulate(row)
                    problem_matches[row["problem"]] = None
                    shard_rows += 1
                    shard_bytes_read += len(row["problem"].encode()) + len(row["generated_solution"].encode())
            if shard_rows != item["rows"] or shard_bytes_read != item["raw_text_bytes"]:
                raise ValueError("Committed output census mismatch")
            restored_rows += shard_rows
        if restored_rows != counts["retained"] or raw_bytes != saved["raw_text_bytes"]:
            raise ValueError("Rebuilt aggregates disagree with checkpoint")
        # Do not delete anything until bindings, source hashes and committed
        # output hashes/census all pass. Only the exact next writer paths qualify.
        for path in orphans:
            path.unlink()
        _sync_directory(staging)
    else:
        staging.mkdir(parents=True)
        _sync_directory(staging.parent)

    def checkpoint() -> None:
        if pending:
            raise ValueError("Cannot checkpoint uncommitted rows")
        payload = {"binding": binding, "next_input": next_input, "next_row": next_row,
                   "counts": dict(counts), "outputs": outputs, "verified_inputs": consumed_inputs,
                   "raw_text_bytes": raw_bytes, "data_complete": data_complete}
        write_json(staging / "checkpoint.json", {"schema": "disjoint_math_sft_checkpoint/v1",
                                                 "payload": payload, "payload_sha256": _checkpoint_digest(payload)})
        _sync_directory(staging)

    def flush() -> None:
        nonlocal pending, pending_bytes
        if pending:
            path = staging / f"part-{len(outputs):05d}.parquet"
            temporary = path.with_suffix(path.suffix + ".tmp")
            pq.write_table(pa.Table.from_pylist(pending, schema=schema), temporary, compression="zstd", row_group_size=128)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            temporary.replace(path)
            _sync_directory(staging)
            outputs.append({"path": path.name, "rows": len(pending), "raw_text_bytes": pending_bytes,
                            "bytes": path.stat().st_size, "sha256": sha256(path)})
            pending, pending_bytes = [], 0
            print(json.dumps({"retained_rows": counts["retained"], "raw_text_bytes": raw_bytes,
                              "output_shards": len(outputs), "counts": dict(counts)}), flush=True)
        checkpoint()

    def paused() -> dict:
        flush()
        return {"paused": True, "checkpoint": str(staging / "checkpoint.json"),
                "raw_text_bytes": raw_bytes, "counts": dict(counts)}

    checkpoint()
    completed_this_invocation = 0
    while next_input < len(inputs) and raw_bytes < target_bytes:
        if should_stop():
            return paused()
        item = inputs[next_input]
        if "rows" not in item:
            item = verify_input(input_dir, item, deadline)
            inputs[next_input] = item
        if len(consumed_inputs) == next_input:
            consumed_inputs.append(item)
        file = pq.ParquetFile(input_dir / item["path"])
        if not input_schema.equals(file.schema_arrow, check_metadata=False):
            raise ValueError("Input schemas differ; metadata must not be silently dropped")
        scanned_rows = 0
        for batch in file.iter_batches(batch_size=128):
            batch_start = scanned_rows
            scanned_rows += batch.num_rows
            if scanned_rows <= next_row:
                continue
            if should_stop():
                return paused()
            for source_row, row in enumerate(batch.to_pylist(), start=batch_start):
                if source_row < next_row:
                    continue
                next_row = source_row + 1
                counts["examined"] += 1
                try:
                    row = normalize_row(row)
                except ValueError:
                    counts["unrecognized_prompt_framing"] += 1
                    continue
                reason = quality_reason(row)
                if reason:
                    counts[reason] += 1
                    continue
                problem, solution = row["problem"], row["generated_solution"]
                if problem not in problem_matches:
                    matches = index.matches(problem)
                    problem_matches[problem] = matches[0].matcher if matches else None
                if problem_matches[problem]:
                    counts[f"protected_{problem_matches[problem]}"] += 1
                    continue
                pair = hashlib.sha256(problem.encode() + b"\x00" + solution.encode()).digest()
                if pair in pairs:
                    counts["duplicate_question_solution"] += 1
                    continue
                row.update(source_shard=item["path"], source_row=source_row, final_answer=terminal_box(solution))
                accumulate(row)
                pending_bytes += len(problem.encode()) + len(solution.encode())
                pending.append(row)
                counts["retained"] += 1
                data_complete = raw_bytes >= target_bytes
                if pending_bytes >= shard_bytes:
                    flush()
                if data_complete:
                    break
            if should_stop():
                return paused()
            if data_complete:
                break
        if next_row == item["rows"]:
            next_input += 1
            next_row = 0
            completed_this_invocation += 1
            data_complete = raw_bytes >= target_bytes or next_input == len(inputs)
            flush()
            if stop_after_input_shards is not None and completed_this_invocation >= stop_after_input_shards:
                return paused()
        if data_complete:
            break
    data_complete = True
    flush()
    if should_stop():
        return paused()
    if not outputs:
        raise ValueError("No retained rows; refusing to publish an empty corpus")
    try:
        audit = audit_output(staging, index, outputs, should_stop=should_stop)
    except PreparationPaused:
        return paused()
    if audit["rows"] != counts["retained"] or audit["raw_text_bytes"] != raw_bytes:
        raise ValueError("Independent audit totals disagree")
    # Revalidate references after the potentially long build, before publication.
    for reference in protection["references"]:
        if sha256(Path(reference["path"])) != reference["sha256"]:
            raise ValueError("Protected reference changed during build")
    if "rl_manifest_sha256" in protection and sha256(rl_manifest) != protection["rl_manifest_sha256"]:
        raise ValueError("RL manifest changed during build")
    if sha256(input_dir / "source_manifest.json") != source_manifest["manifest_sha256"]:
        raise ValueError("Source manifest changed during build")
    if _checkpoint_bindings(input_dir, output_dir, rl_manifest, evaluation_paths,
                            source_manifest, protection, target_bytes, shard_bytes) != binding:
        raise ValueError("Preparation code/configuration changed during build")
    if should_stop():
        return paused()
    manifest = {"schema": "disjoint_math_sft/v1", "target_bytes": target_bytes, "target_met": raw_bytes >= target_bytes,
                "raw_text_bytes": raw_bytes, "size_definition": "UTF-8 bytes of problem plus generated_solution",
                "source": source_manifest, "inputs": consumed_inputs, "outputs": outputs, "protection": protection,
                "counts": dict(counts), "unique_questions": len(repetitions),
                "solutions_per_question_histogram": dict(Counter(repetitions.values())),
                "character_length_histogram": dict(histogram), "problem_sources": dict(sources),
                "generation_models": dict(models), "problem_types": dict(answer_origins),
                "expected_answer_provenance": "Upstream expected_answer and problem_type are retained, not independently verified. Source card distinguishes extracted answers, majority-vote answers, and converted proofs; per-row problem_type is the upstream label, not a new correctness certification.",
                "answer_policy": "Terminal boxed answer agrees textually after limited formatting normalization with upstream expected_answer; not independent correctness verification.",
                "solution_normalization": "Remove exactly one initial think-open and one think-close marker; retain all content. Original trace reconstructible from pinned source_shard/source_row, checked by source_solution_sha256.",
                "problem_normalization": "Repository strip_math_prompt_framing; original reconstructible from source_shard/source_row and source_problem_sha256.",
                "ordering": "lexicographic input shard path, original row order; first passing rows to target",
                "truncation": "none", "builder_sha256": sha256(Path(__file__)), "audit": audit}
    write_json(staging / "audit.json", audit)
    write_json(staging / "manifest.json", manifest)
    write_json(staging / "COMPLETE.json", {"manifest_sha256": sha256(staging / "manifest.json"),
                                          "audit_sha256": sha256(staging / "audit.json"), "target_met": manifest["target_met"]})
    _sync_directory(staging)
    staging.rename(output_dir)
    _sync_directory(output_dir.parent)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rl-manifest", type=Path, default=DEFAULT_RL)
    parser.add_argument("--target-bytes", type=int, default=50_000_000_000)
    parser.add_argument("--shard-bytes", type=int, default=512_000_000)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Explicitly resume the matching durable .building checkpoint")
    parser.add_argument("--stop-after-input-shards", type=int,
                        help="Pause after this many input-file boundaries in this invocation (exit 75)")
    parser.add_argument("--wait-for-input-seconds", type=float, default=0,
                        help="Optional overall bounded deadline for atomically arriving pinned input shards; default fails closed immediately")
    args = parser.parse_args()
    if args.audit_only:
        manifest = json.loads((args.output_dir / "manifest.json").read_text())
        complete = json.loads((args.output_dir / "COMPLETE.json").read_text())
        if sha256(args.output_dir / "manifest.json") != complete["manifest_sha256"]:
            raise ValueError("Completion manifest hash mismatch")
        if sha256(args.output_dir / "audit.json") != complete["audit_sha256"]:
            raise ValueError("Completion audit hash mismatch")
        protection = manifest["protection"]
        for reference in protection["references"]:
            if sha256(Path(reference["path"])) != reference["sha256"]:
                raise ValueError("Protected reference hash changed")
        rl_path = Path(protection["rl_manifest"])
        if sha256(rl_path) != protection["rl_manifest_sha256"]:
            raise ValueError("RL manifest hash changed")
        eval_paths = tuple(Path(r["path"]) for r in protection["references"] if r["kind"] == "evaluation_or_reserved")
        index, _ = load_references(rl_path, eval_paths)
        audit = audit_output(args.output_dir, index, manifest["outputs"])
        if audit != manifest["audit"]:
            raise ValueError("Independent audit disagrees with published audit")
        print(json.dumps(audit, indent=2))
        return 0
    if args.input_dir is None:
        parser.error("--input-dir is required for building")
    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    previous_handlers = {kind: signal.signal(kind, request_stop) for kind in (signal.SIGINT, signal.SIGTERM)}
    try:
        manifest = build(args.input_dir, args.output_dir, args.rl_manifest, args.target_bytes,
                         tuple(ROOT / path for path in EVALUATION_PATHS), args.shard_bytes, args.wait_for_input_seconds,
                         resume=args.resume, stop_after_input_shards=args.stop_after_input_shards,
                         should_stop=lambda: stopping)
    finally:
        for kind, handler in previous_handlers.items():
            signal.signal(kind, handler)
    if manifest.get("paused"):
        print(json.dumps(manifest))
        return 75
    print(json.dumps({"output": str(args.output_dir), "raw_text_bytes": manifest["raw_text_bytes"],
                      "target_met": manifest["target_met"]}))
    return 0 if manifest["target_met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
