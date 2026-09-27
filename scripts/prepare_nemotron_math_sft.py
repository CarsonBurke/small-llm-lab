#!/usr/bin/env python3
"""Add conservatively screened Nemotron COT math to a frozen disjoint archive.

CPU-only preparation. Published-answer verification is an upstream claim;
local screening checks textual answer consistency and question overlap only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import platform
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
from scripts import prepare_disjoint_math_sft as common

REPO = "nvidia/Nemotron-SFT-Math-v4"
REVISION = "84d42ad0cb960f07f951b9baa9ed2b46a5a18c66"
LICENSES = {"AoPS": "cc-by-4.0", "Math StackExchange": "cc-by-sa-4.0"}
GENERIC_SYSTEM_PROMPTS = {"", "you are a helpful assistant.", "you are a helpful ai assistant."}
ATTRIBUTION = ("uuid", "license", "url", "user_url", "username", "source", "dataset", "subset")
SCHEMA = pa.schema([
    *(pa.field(name, pa.string()) for name in (
        "problem", "generated_solution", "expected_answer", "final_answer", "generation_model",
        "problem_source", "inference_mode", *ATTRIBUTION, "source_metadata", "source_system_messages",
        "source_shard", "source_row_sha256")),
    pa.field("source_row", pa.int64()),
])


class PreparationInterrupted(Exception):
    """A durable checkpoint was saved; the CLI exits with temporary status 75."""


def json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def pair_hash(problem: str, solution: str) -> bytes:
    return hashlib.sha256(problem.encode() + b"\x00" + solution.encode()).digest()


def quality_reason(row: dict) -> str | None:
    if row.get("inference_mode") != "cot" or row.get("subset") != "cot":
        return "non_cot"
    if row.get("source") not in LICENSES or row.get("license") != LICENSES[row["source"]]:
        return "unknown_source_or_license"
    if row.get("dataset") != "Nemotron-SFT-Math-v4":
        return "unknown_dataset"
    if not isinstance(row.get("uuid"), str) or not row["uuid"].strip():
        return "missing_uuid"
    for name in ("problem", "generated_solution", "expected_answer"):
        value = row.get(name)
        if not isinstance(value, str) or not value.strip():
            return "empty_or_malformed_text"
        if "\x00" in value or "\ufffd" in value or common.RESERVED.search(value):
            return "invalid_text_or_control_artifact"
    final = common.terminal_box(row["generated_solution"])
    if final is None:
        return "no_terminal_boxed_answer"
    if common.answer_key(final) != common.answer_key(row["expected_answer"]):
        return "boxed_expected_text_disagreement"
    if row.get("final_answer") != final:
        return "final_answer_mismatch"
    return None


def adapt(row: dict, shard: str, position: int) -> tuple[dict | None, str | None]:
    if row.get("subset") != "cot":
        return None, "non_cot"
    if row.get("tools"):
        return None, "tools_present"
    messages = row.get("messages")
    if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages):
        return None, "malformed_messages"
    roles = [message.get("role") for message in messages]
    if roles not in (["user", "assistant"], ["system", "user", "assistant"]):
        return None, "non_single_turn"
    if any(message.get("tool_calls") or message.get("tool_call_id") or message.get("name") for message in messages):
        return None, "tool_message_artifact"
    if any(message.get("reasoning_content") for message in messages[:-1]):
        return None, "nonassistant_reasoning"
    if roles[0] == "system":
        system_content = messages[0].get("content")
        if not isinstance(system_content, str) or common.whitespace_key(system_content) not in GENERIC_SYSTEM_PROMPTS:
            return None, "unrecognized_system_instructions"
    user, assistant = messages[-2:]
    if not isinstance(row.get("problem"), str) or not isinstance(user.get("content"), str):
        return None, "malformed_problem"
    try:
        problem = common.strip_math_prompt_framing(row["problem"])[0]
        prompt = common.strip_math_prompt_framing(user["content"])[0]
    except ValueError:
        return None, "unrecognized_prompt_framing"
    if common.whitespace_key(problem) != common.whitespace_key(prompt):
        return None, "user_problem_disagreement"
    content = assistant.get("content")
    reasoning = assistant.get("reasoning_content")
    if not isinstance(content, str) or not content.strip() or (reasoning is not None and not isinstance(reasoning, str)):
        return None, "malformed_assistant"
    if reasoning:
        if common.RESERVED.search(reasoning + content):
            return None, "ambiguous_reasoning_or_control_artifact"
        solution = reasoning + "\n\n" + content
    else:
        solution = common.normalize_row({"problem": problem, "generated_solution": content})["generated_solution"]
    output = {"problem": problem, "generated_solution": solution, "expected_answer": row.get("expected_answer"),
              "final_answer": common.terminal_box(solution), "generation_model": "DeepSeek-V4-Pro (high)",
              "problem_source": row.get("source"), "inference_mode": "cot",
              **{name: row.get(name) for name in ATTRIBUTION},
              "source_metadata": json_bytes({key: value for key, value in row.items() if key != "messages"}).decode(),
              "source_system_messages": json_bytes(messages[:-2]).decode(),
              "source_row_sha256": hashlib.sha256(json_bytes(row)).hexdigest(),
              "source_shard": shard, "source_row": position}
    reason = quality_reason(output)
    return (None, reason) if reason else (output, None)


def wait_for(path: Path, deadline: float | None, stop_requested=lambda: False) -> None:
    while not path.is_file() and deadline is not None and time.monotonic() < deadline:
        if stop_requested():
            raise PreparationInterrupted("Stopped waiting for inputs; existing committed data is unchanged")
        print(json.dumps({"waiting_for": str(path)}), flush=True)
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    if not path.is_file():
        raise FileNotFoundError(path)


def source_inventory(directory: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((directory / "source_manifest.json").read_text())
    if manifest.get("repo_id") != REPO or manifest.get("revision") != REVISION:
        raise ValueError("Unexpected source repository or revision")
    files = sorted(manifest["files"], key=lambda item: item["path"])
    declared = {item["path"] for item in files}
    if not files or len(declared) != len(files):
        raise ValueError("Missing or duplicate input declarations")
    for item in files:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".parquet":
            raise ValueError("Unsafe source path")
        if not isinstance(item.get("bytes"), int) or item["bytes"] <= 0 or not re.fullmatch("[0-9a-f]{64}", item.get("sha256", "")):
            raise ValueError("Invalid source size/hash")
    if {str(path.relative_to(directory)) for path in directory.rglob("*.parquet")} - declared:
        raise ValueError("Undeclared input shards")
    return {**manifest, "manifest_sha256": common.sha256(directory / "source_manifest.json")}, files


def load_base(manifest_path: Path | None, deadline: float | None,
              stop_requested=lambda: False) -> tuple[dict | None, set[bytes]]:
    if manifest_path is None:
        return None, set()
    wait_for(manifest_path.parent / "COMPLETE.json", deadline, stop_requested)
    manifest = json.loads(manifest_path.read_text())
    complete = json.loads((manifest_path.parent / "COMPLETE.json").read_text())
    digest = common.sha256(manifest_path)
    if digest != complete["manifest_sha256"] or common.sha256(manifest_path.parent / "audit.json") != complete["audit_sha256"]:
        raise ValueError("Base completion hash mismatch")
    if manifest.get("schema") != "disjoint_math_sft/v1":
        raise ValueError("Expected frozen OpenMathReasoning base manifest")
    if {item["path"] for item in manifest["outputs"]} != {path.name for path in manifest_path.parent.glob("*.parquet")}:
        raise ValueError("Base output shard set mismatch")
    pairs = set()
    actual_bytes = rows = 0
    for item in manifest["outputs"]:
        path = manifest_path.parent / item["path"]
        if common.sha256(path) != item["sha256"]:
            raise ValueError("Base output hash mismatch")
        shard_rows = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=["problem", "generated_solution"]):
            if stop_requested():
                raise PreparationInterrupted("Stopped during base preflight; existing committed data is unchanged")
            for row in batch.to_pylist():
                pairs.add(pair_hash(row["problem"], row["generated_solution"]))
                actual_bytes += len(row["problem"].encode()) + len(row["generated_solution"].encode())
                rows += 1
                shard_rows += 1
        if shard_rows != item["rows"]:
            raise ValueError("Base output rows mismatch")
    if actual_bytes != manifest["raw_text_bytes"] or len(pairs) != rows:
        raise ValueError("Base byte count or dedup invariant mismatch")
    return {"manifest_path": str(manifest_path.resolve()), "manifest_sha256": digest,
            "raw_text_bytes": actual_bytes, "rows": rows, "protection": manifest["protection"]}, pairs


def audit_output(directory: Path, records: list[dict], index, base_pairs: set[bytes], stop_requested=lambda: False) -> dict:
    if {item["path"] for item in records} != {path.name for path in directory.glob("*.parquet")}:
        raise ValueError("Output shard set mismatch")
    seen_pairs, questions = set(), set()
    rows = raw_bytes = 0
    for item in records:
        path = directory / item["path"]
        if common.sha256(path) != item["sha256"]:
            raise ValueError("Output hash mismatch")
        count = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            if stop_requested():
                raise PreparationInterrupted("Stopped during final audit; retained shards remain checkpointed")
            for row in batch.to_pylist():
                reason = quality_reason(row)
                if reason:
                    raise ValueError(f"Output quality audit: {reason}")
                pair = pair_hash(row["problem"], row["generated_solution"])
                if pair in seen_pairs or pair in base_pairs:
                    raise ValueError("Duplicate retained pair within supplement or base")
                seen_pairs.add(pair)
                if row["problem"] not in questions:
                    if index.matches(row["problem"]):
                        raise ValueError("Protected-question overlap in output")
                    questions.add(row["problem"])
                rows += 1
                count += 1
                raw_bytes += len(row["problem"].encode()) + len(row["generated_solution"].encode())
        if count != item["rows"]:
            raise ValueError("Output row count mismatch")
    return {"rows": rows, "raw_text_bytes": raw_bytes, "unique_questions": len(questions),
            "protected_question_matches": 0, "duplicate_pairs_within_or_against_base": 0,
            "method": "Independent full retained-parquet rescan with fresh question cache and original base-pair set",
            "limitations": "Question-level only; unrelated protected questions embedded solely in solutions and semantic paraphrases are not ruled out. Mathematical correctness is not locally verified."}


def build(input_dir: Path, output_dir: Path, rl_manifest: Path, target_bytes: int,
          base_manifest: Path | None = None, wait_seconds: float = 0, shard_bytes: int = 512_000_000,
          *, resume: bool = False, stop_after_input_shards: int | None = None,
          stop_requested=lambda: False) -> dict:
    if target_bytes <= 0 or shard_bytes <= 0 or wait_seconds < 0:
        raise ValueError("Invalid byte budget or wait duration")
    staging = output_dir.with_name(output_dir.name + ".building")
    if output_dir.exists() or (staging.exists() and not resume):
        raise FileExistsError("Output or staging exists; use explicit --resume for staging")
    if resume and not (staging / "checkpoint.json").is_file():
        raise FileNotFoundError("No durable checkpoint to resume")
    if resume and (staging.is_symlink() or (staging / "checkpoint.json").is_symlink()):
        raise ValueError("Resume requires a regular staging directory and checkpoint")
    if stop_after_input_shards is not None and stop_after_input_shards < 1:
        raise ValueError("stop-after-input-shards must be positive")
    deadline = time.monotonic() + wait_seconds if wait_seconds else None
    source, files = source_inventory(input_dir)
    base, base_pairs = load_base(base_manifest, deadline, stop_requested)
    target = max(0, target_bytes - (base["raw_text_bytes"] if base else 0))
    index, protection = common.load_references(rl_manifest, tuple(ROOT / path for path in common.EVALUATION_PATHS))
    if base and base["protection"] != protection:
        raise ValueError("Base and supplement protection sets differ")
    staging.mkdir(parents=True, exist_ok=resume)
    pairs, cache, repetition = set(), {}, Counter()
    counts, histogram, sources, licenses = Counter({"retained": 0, "examined": 0}), Counter(), Counter(), Counter()
    outputs, inputs, pending = [], [], []
    raw_bytes = pending_bytes = 0
    cursor = {"input_index": 0, "next_row": 0}
    bindings = {"source": source, "input_directory": str(input_dir.resolve()),
                "output_directory": str(output_dir.resolve()), "rl_manifest": str(rl_manifest.resolve()),
                "base": base, "protection": protection, "target_bytes": target_bytes,
                "shard_bytes": shard_bytes,
                "runtime": {"python": platform.python_version(), "pyarrow": pa.__version__},
                "code": {str(path.relative_to(ROOT)): common.sha256(path) for path in (
                    Path(__file__).resolve(), Path(common.__file__).resolve(),
                    ROOT / "postraining/problem_overlap.py", ROOT / "postraining/math_prompt.py",
                    ROOT / "postraining/choice_prompt.py", ROOT / "postraining/decontaminate.py",
                    ROOT / "postraining/problem_registry.py", ROOT / "postraining/core.py")}}

    def checkpoint() -> None:
        if pending:
            raise RuntimeError("Checkpoint cannot skip uncommitted rows")
        state = {
            "schema": "nemotron_math_sft_checkpoint/v1", "bindings": bindings,
            "cursor": cursor, "counts": dict(counts), "outputs": outputs,
            "inputs": inputs, "raw_text_bytes": raw_bytes}
        state["state_sha256"] = hashlib.sha256(json_bytes(state)).hexdigest()
        common.write_json(staging / "checkpoint.json", state)
        # Ensure the renamed checkpoint and committed shards survive a crash.
        descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    if resume:
        saved = json.loads((staging / "checkpoint.json").read_text())
        saved_digest = saved.pop("state_sha256", None)
        if saved_digest != hashlib.sha256(json_bytes(saved)).hexdigest():
            raise ValueError("Checkpoint state checksum mismatch")
        if saved.get("schema") != "nemotron_math_sft_checkpoint/v1" or saved.get("bindings") != bindings:
            raise ValueError("Checkpoint bindings changed: source/base/policy/code/config")
        outputs, inputs = saved["outputs"], saved["inputs"]
        cursor, counts = saved["cursor"], Counter(saved["counts"])
        if not (0 <= cursor["input_index"] <= len(files)) or cursor["next_row"] < 0:
            raise ValueError("Invalid checkpoint cursor")
        if cursor["input_index"] == len(files) and cursor["next_row"]:
            raise ValueError("Invalid final checkpoint cursor")
        expected_input_count = cursor["input_index"] + bool(cursor["next_row"])
        if len(inputs) != expected_input_count:
            raise ValueError("Checkpoint input/cursor count mismatch")
        if [item["path"] for item in inputs] != [item["path"] for item in files[:len(inputs)]]:
            raise ValueError("Checkpoint inputs are not the source prefix")
        examined = sum(item["rows"] for item in inputs[:cursor["input_index"]]) + cursor["next_row"]
        if examined != counts["examined"] or sum(value for key, value in counts.items() if key != "examined") != examined:
            raise ValueError("Checkpoint cursor/counts mismatch")
        for item in inputs:
            declared = next((entry for entry in files if entry["path"] == item["path"]), None)
            if declared is None or any(item[key] != declared[key] for key in ("sha256", "bytes")):
                raise ValueError("Checkpoint input differs from source manifest")
            source_path = input_dir / item["path"]
            if source_path.stat().st_size != item["bytes"] or common.sha256(source_path) != item["sha256"]:
                raise ValueError("Previously consumed input changed")
        expected_names = {"checkpoint.json"}
        for number, item in enumerate(outputs):
            if item["path"] != f"part-{number:05d}.parquet":
                raise ValueError("Checkpoint output sequence is invalid")
            path = staging / item["path"]
            expected_names.add(path.name)
            if path.is_symlink() or path.stat().st_size != item["bytes"] or common.sha256(path) != item["sha256"]:
                raise ValueError("Committed checkpoint output changed")
            rows = size = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    if quality_reason(row):
                        raise ValueError("Committed checkpoint row failed quality")
                    problem, solution = row["problem"], row["generated_solution"]
                    pair = pair_hash(problem, solution)
                    if pair in pairs or pair in base_pairs:
                        raise ValueError("Duplicate committed checkpoint pair")
                    pairs.add(pair)
                    repetition[problem] += 1
                    histogram[common.length_bucket(len(problem) + len(solution))] += 1
                    sources[row["source"]] += 1
                    licenses[row["license"]] += 1
                    rows += 1
                    size += len(problem.encode()) + len(solution.encode())
            if rows != item["rows"] or size != item["raw_text_bytes"]:
                raise ValueError("Checkpoint output census mismatch")
            raw_bytes += size
        if raw_bytes != saved["raw_text_bytes"] or len(pairs) != counts["retained"]:
            raise ValueError("Checkpoint totals mismatch")
        allowed_orphans = {f"part-{len(outputs):05d}.parquet", f"part-{len(outputs):05d}.parquet.tmp",
                           "checkpoint.json.tmp"}
        if raw_bytes >= target or cursor["input_index"] == len(files):
            allowed_orphans.update({name + suffix for name in ("audit.json", "manifest.json", "COMPLETE.json")
                                    for suffix in ("", ".tmp")})
        extras = [path for path in staging.iterdir() if path.name not in expected_names]
        if any(not path.is_file() or path.is_symlink() or path.name not in allowed_orphans for path in extras):
            raise ValueError("Unexpected file in checkpoint staging directory")
        for path in extras:
            path.unlink()
    else:
        checkpoint()
    if stop_requested():
        raise PreparationInterrupted("Stopped before next batch; durable checkpoint available")

    def flush() -> None:
        nonlocal pending, pending_bytes
        if not pending:
            return
        path = staging / f"part-{len(outputs):05d}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(pending, schema=SCHEMA), temporary, compression="zstd", row_group_size=128)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
        outputs.append({"path": path.name, "rows": len(pending), "raw_text_bytes": pending_bytes,
                        "bytes": path.stat().st_size, "sha256": common.sha256(path)})
        pending, pending_bytes = [], 0
        checkpoint()
        print(json.dumps({"counts": dict(counts), "raw_text_bytes": raw_bytes,
                          "remaining_target_bytes": max(0, target - raw_bytes)}), flush=True)

    completed_this_invocation = 0
    for input_index in range(cursor["input_index"], len(files)):
        item = files[input_index]
        if raw_bytes >= target:
            break
        path = input_dir / item["path"]
        wait_for(path, deadline, stop_requested)
        if path.stat().st_size != item["bytes"] or common.sha256(path) != item["sha256"]:
            raise ValueError("Input size/hash mismatch")
        file = pq.ParquetFile(path)
        if not {"problem", "messages", "expected_answer", *ATTRIBUTION} <= set(file.schema_arrow.names):
            raise ValueError("Missing required source columns")
        verified = {**item, "rows": file.metadata.num_rows}
        if input_index < len(inputs):
            if inputs[input_index] != verified:
                raise ValueError("Checkpoint verified-input census changed")
        else:
            inputs.append(verified)
        start_row = cursor["next_row"] if input_index == cursor["input_index"] else 0
        if start_row > file.metadata.num_rows:
            raise ValueError("Checkpoint cursor exceeds input rows")
        position = -1
        for batch in file.iter_batches(batch_size=128):
            for original in batch.to_pylist():
                position += 1
                if position < start_row:
                    continue
                cursor.update(input_index=input_index, next_row=position + 1)
                counts["examined"] += 1
                row, reason = adapt(original, item["path"], position)
                if reason:
                    counts[reason] += 1
                    continue
                problem, solution = row["problem"], row["generated_solution"]
                if problem not in cache:
                    matches = index.matches(problem)
                    cache[problem] = matches[0].matcher if matches else None
                if cache[problem]:
                    counts[f"protected_{cache[problem]}"] += 1
                    continue
                pair = pair_hash(problem, solution)
                if pair in pairs or pair in base_pairs:
                    counts["duplicate_pair_within_or_against_base"] += 1
                    continue
                pairs.add(pair)
                size = len(problem.encode()) + len(solution.encode())
                raw_bytes += size
                pending_bytes += size
                pending.append(row)
                counts["retained"] += 1
                repetition[problem] += 1
                histogram[common.length_bucket(len(problem) + len(solution))] += 1
                sources[row["source"]] += 1
                licenses[row["license"]] += 1
                if pending_bytes >= shard_bytes:
                    flush()
                if raw_bytes >= target:
                    break
            if raw_bytes >= target:
                break
            if stop_requested():
                flush()
                checkpoint()
                raise PreparationInterrupted("Stopped at batch boundary; resume the durable checkpoint")
        if position + 1 >= file.metadata.num_rows:
            cursor.update(input_index=input_index + 1, next_row=0)
            completed_this_invocation += 1
        flush()
        checkpoint()
        if stop_requested() or (stop_after_input_shards is not None and completed_this_invocation >= stop_after_input_shards):
            raise PreparationInterrupted("Stopped at input boundary; resume the durable checkpoint")
    flush()
    checkpoint()
    if not outputs and target:
        raise ValueError("No retained supplement rows")
    audit = audit_output(staging, outputs, index, base_pairs, stop_requested)
    if stop_requested():
        raise PreparationInterrupted("Stopped after final audit; durable checkpoint available")
    if audit["raw_text_bytes"] != raw_bytes or audit["rows"] != counts["retained"]:
        raise ValueError("Independent audit totals disagree")
    for reference in protection["references"]:
        if common.sha256(Path(reference["path"])) != reference["sha256"]:
            raise ValueError("Protected reference changed")
    if common.sha256(rl_manifest) != protection["rl_manifest_sha256"]:
        raise ValueError("RL manifest changed")
    if common.sha256(input_dir / "source_manifest.json") != source["manifest_sha256"]:
        raise ValueError("Source manifest changed")
    if base and common.sha256(Path(base["manifest_path"])) != base["manifest_sha256"]:
        raise ValueError("Base manifest changed")
    if any(common.sha256(ROOT / relative) != digest for relative, digest in bindings["code"].items()):
        raise ValueError("Preparation code changed during build; do not publish mixed-policy output")
    combined = raw_bytes + (base["raw_text_bytes"] if base else 0)
    manifest = {"schema": "disjoint_nemotron_math_sft/v1", "source": source, "inputs": inputs, "outputs": outputs,
                "base": base, "target_bytes": target_bytes, "supplement_target_bytes": target,
                "raw_text_bytes": raw_bytes, "combined_raw_text_bytes": combined, "target_met": combined >= target_bytes,
                "size_definition": "UTF-8 bytes of normalized problem plus complete generated_solution, once",
                "counts": dict(counts), "unique_questions": len(repetition),
                "solutions_per_question_histogram": dict(Counter(repetition.values())),
                "character_length_histogram": dict(histogram), "sources": dict(sources), "licenses": dict(licenses),
                "protection": protection, "audit": audit, "builder_sha256": common.sha256(Path(__file__)),
                "common_builder_sha256": common.sha256(Path(common.__file__)),
                "ordering": "Lexicographic shard then source row; no truncation",
                "answer_provenance": "Original expected_answer retained; upstream card claims verified reference agreement. Locally requires terminal boxed textual agreement after limited formatting normalization, not independent mathematical verification.",
                "license_attribution": "Per-row license/source/url/user_url/username retained; CC BY 4.0 AoPS and CC BY-SA 4.0 Math StackExchange. Original metadata retained and exact full messages reconstructible from source_shard/source_row and canonical-JSON source_row_sha256.",
                "solution_normalization": "Complete assistant reasoning_content plus two newlines plus content; if no separate reasoning, only valid outer think markers are removed. Original source remains pinned."}
    common.write_json(staging / "audit.json", audit)
    common.write_json(staging / "manifest.json", manifest)
    common.write_json(staging / "COMPLETE.json", {"manifest_sha256": common.sha256(staging / "manifest.json"),
                                                "audit_sha256": common.sha256(staging / "audit.json"), "target_met": manifest["target_met"]})
    staging.rename(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--rl-manifest", type=Path, default=common.DEFAULT_RL)
    parser.add_argument("--target-bytes", type=int, default=50_000_000_000)
    parser.add_argument("--shard-bytes", type=int, default=512_000_000)
    parser.add_argument("--wait-for-input-seconds", type=float, default=0)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-input-shards", type=int)
    args = parser.parse_args()
    if args.audit_only:
        manifest_path = args.output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        complete = json.loads((args.output_dir / "COMPLETE.json").read_text())
        if common.sha256(manifest_path) != complete["manifest_sha256"] or common.sha256(args.output_dir / "audit.json") != complete["audit_sha256"]:
            raise ValueError("Completion hash mismatch")
        base_path = Path(manifest["base"]["manifest_path"]) if manifest["base"] else None
        base, base_pairs = load_base(base_path, None)
        if base != manifest["base"]:
            raise ValueError("Base manifest changed")
        protection = manifest["protection"]
        for reference in protection["references"]:
            if common.sha256(Path(reference["path"])) != reference["sha256"]:
                raise ValueError("Protected reference changed")
        rl_path = Path(protection["rl_manifest"])
        if common.sha256(rl_path) != protection["rl_manifest_sha256"]:
            raise ValueError("RL manifest changed")
        eval_paths = tuple(Path(item["path"]) for item in protection["references"] if item["kind"] == "evaluation_or_reserved")
        index, fresh_protection = common.load_references(rl_path, eval_paths)
        if fresh_protection != protection:
            raise ValueError("Protection policy changed")
        audit = audit_output(args.output_dir, manifest["outputs"], index, base_pairs)
        if audit != manifest["audit"]:
            raise ValueError("Independent audit differs from published audit")
        print(json.dumps(audit, indent=2))
        return 0
    if args.input_dir is None:
        parser.error("--input-dir is required for building")
    stopping = False
    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        manifest = build(args.input_dir, args.output_dir, args.rl_manifest, args.target_bytes, args.base_manifest,
                         args.wait_for_input_seconds, args.shard_bytes, resume=args.resume,
                         stop_after_input_shards=args.stop_after_input_shards, stop_requested=lambda: stopping)
    except PreparationInterrupted as exc:
        print(str(exc), flush=True)
        return 75
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(json.dumps({"output": str(args.output_dir), "raw_text_bytes": manifest["raw_text_bytes"],
                      "combined_raw_text_bytes": manifest["combined_raw_text_bytes"], "target_met": manifest["target_met"]}))
    return 0 if manifest["target_met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
