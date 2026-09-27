from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import prepare_nemotron_math_sft as builder
from postraining.problem_overlap import ProblemOverlapIndex


def source_row(problem="Compute 17+29."):
    return {"uuid": "fixture-1", "problem": problem, "expected_answer": "46",
            "messages": [{"role": "user", "content": problem + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
                          "reasoning_content": None, "tool_calls": None, "name": None, "tool_call_id": None},
                         {"role": "assistant", "content": r"The answer is \boxed{46}",
                          "reasoning_content": "Reason through the arithmetic.", "tool_calls": None,
                          "name": None, "tool_call_id": None}],
            "tools": [], "source": "AoPS", "license": "cc-by-4.0", "url": None, "user_url": None,
            "username": None, "dataset": "Nemotron-SFT-Math-v4", "subset": "cot", "used_in": [], "metadata": []}


def write_source(tmp_path, rows):
    directory = tmp_path / "source"
    directory.mkdir()
    path = directory / "part.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    builder.common.write_json(directory / "source_manifest.json", {
        "repo_id": builder.REPO, "revision": builder.REVISION,
        "files": [{"path": path.name, "bytes": path.stat().st_size, "sha256": builder.common.sha256(path)}],
    })
    return directory


def test_separate_reasoning_preserved_and_attribution_recorded():
    original = source_row()
    original["messages"][1]["reasoning_content"] = "x" * 100_000
    result, reason = builder.adapt(original, "part.parquet", 4)
    assert reason is None
    assert result["generated_solution"].startswith("x" * 100_000 + "\n\n")
    assert result["final_answer"] == "46"
    assert result["source_row"] == 4
    assert result["license"] == "cc-by-4.0"
    assert len(result["source_row_sha256"]) == 64
    assert json.loads(result["source_metadata"])["uuid"] == "fixture-1"
    assert "messages" not in json.loads(result["source_metadata"])


@pytest.mark.parametrize("change,reason", [
    ({"subset": "tir"}, "non_cot"),
    ({"tools": [{"function": "python"}]}, "tools_present"),
    ({"license": "unknown"}, "unknown_source_or_license"),
    ({"expected_answer": "47"}, "boxed_expected_text_disagreement"),
    ({"source": "Unknown"}, "unknown_source_or_license"),
])
def test_fail_closed_policy(change, reason):
    result, actual = builder.adapt({**source_row(), **change}, "part.parquet", 0)
    assert result is None and actual == reason


def test_tool_turn_and_prompt_disagreement_rejected():
    row = source_row()
    row["messages"].insert(1, {"role": "tool", "content": "46"})
    assert builder.adapt(row, "part", 0)[1] == "non_single_turn"
    row = source_row()
    row["messages"][0]["content"] = "A different task"
    assert builder.adapt(row, "part", 0)[1] == "user_problem_disagreement"


def test_only_generic_system_prompts_and_no_user_reasoning_can_be_dropped():
    row = source_row()
    row["messages"].insert(0, {"role": "system", "content": "You are a helpful assistant."})
    assert builder.adapt(row, "part", 0)[1] is None
    row["messages"][0]["content"] = "Assume all arithmetic is modulo 7."
    assert builder.adapt(row, "part", 0)[1] == "unrecognized_system_instructions"
    row = source_row()
    row["messages"][0]["reasoning_content"] = "Assume all arithmetic is modulo 7."
    assert builder.adapt(row, "part", 0)[1] == "nonassistant_reasoning"


def test_expected_remains_independent_of_extracted_final():
    result, _ = builder.adapt(source_row(), "part", 0)
    assert builder.quality_reason({**result, "final_answer": "47"}) == "final_answer_mismatch"
    assert builder.quality_reason({**result, "expected_answer": "47"}) == "boxed_expected_text_disagreement"


def test_source_revision_and_hash_validation(tmp_path):
    directory = write_source(tmp_path, [source_row()])
    manifest, files = builder.source_inventory(directory)
    assert manifest["revision"] == builder.REVISION
    assert len(files) == 1
    path = directory / "source_manifest.json"
    changed = json.loads(path.read_text())
    changed["revision"] = "wrong"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="revision"):
        builder.source_inventory(directory)


def test_build_and_full_audit_remove_duplicates_and_protected_questions(tmp_path, monkeypatch):
    directory = write_source(tmp_path, [source_row(), source_row(), source_row("Protected problem")])
    rl_manifest = tmp_path / "rl.json"
    rl_manifest.write_text("{}")
    protection = {"references": [], "rl_manifest_sha256": builder.common.sha256(rl_manifest)}
    index = ProblemOverlapIndex(["Protected problem"])
    monkeypatch.setattr(builder.common, "load_references", lambda *_: (index, protection))
    output = tmp_path / "output"
    manifest = builder.build(directory, output, rl_manifest, 10_000)
    assert manifest["counts"]["retained"] == 1
    assert manifest["counts"]["protected_whitespace"] == 1
    assert manifest["counts"]["duplicate_pair_within_or_against_base"] == 1
    assert manifest["target_met"] is False
    assert (output / "COMPLETE.json").exists()
    assert manifest["audit"]["protected_question_matches"] == 0
    with pytest.raises(FileExistsError):
        builder.build(directory, output, rl_manifest, 10_000)
    retained = pq.read_table(output / manifest["outputs"][0]["path"]).to_pylist()[0]
    base_pairs = {builder.pair_hash(retained["problem"], retained["generated_solution"])}
    with pytest.raises(ValueError, match="Duplicate retained"):
        builder.audit_output(output, manifest["outputs"], index, base_pairs)


def test_base_hash_census_and_pair_dedup(tmp_path):
    directory = tmp_path / "base"
    directory.mkdir()
    path = directory / "part.parquet"
    pq.write_table(pa.Table.from_pylist([{"problem": "q", "generated_solution": "s"}]), path)
    manifest = {"schema": "disjoint_math_sft/v1", "raw_text_bytes": 2, "protection": {},
                "outputs": [{"path": path.name, "sha256": builder.common.sha256(path), "rows": 1}]}
    builder.common.write_json(directory / "manifest.json", manifest)
    builder.common.write_json(directory / "audit.json", {})
    builder.common.write_json(directory / "COMPLETE.json", {
        "manifest_sha256": builder.common.sha256(directory / "manifest.json"),
        "audit_sha256": builder.common.sha256(directory / "audit.json"),
    })
    base, pairs = builder.load_base(directory / "manifest.json", None)
    assert base["raw_text_bytes"] == 2
    assert pairs == {builder.pair_hash("q", "s")}
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Base output hash"):
        builder.load_base(directory / "manifest.json", None)


def checkpoint_fixture(tmp_path, monkeypatch, rows=None):
    directory = write_source(tmp_path, rows or [source_row(), source_row(), source_row("Other question")])
    rl = tmp_path / "rl.json"
    rl.write_text("{}")
    protection = {"references": [], "rl_manifest_sha256": builder.common.sha256(rl)}
    monkeypatch.setattr(builder.common, "load_references", lambda *_: (ProblemOverlapIndex([]), protection))
    return directory, rl


def retained_rows(directory, manifest):
    return [row for item in manifest["outputs"]
            for row in pq.read_table(directory / item["path"]).to_pylist()]


def test_interrupted_resume_matches_uninterrupted(tmp_path, monkeypatch):
    directory, rl = checkpoint_fixture(tmp_path, monkeypatch)
    second = directory / "second.parquet"
    pq.write_table(pa.Table.from_pylist([source_row(), source_row("Last question")]), second)
    source_manifest = directory / "source_manifest.json"
    inventory = json.loads(source_manifest.read_text())
    inventory["files"].append({"path": second.name, "bytes": second.stat().st_size,
                               "sha256": builder.common.sha256(second)})
    builder.common.write_json(source_manifest, inventory)
    uninterrupted = tmp_path / "clean"
    clean = builder.build(directory, uninterrupted, rl, 100_000, shard_bytes=100)
    output = tmp_path / "resumed"
    with pytest.raises(builder.PreparationInterrupted):
        builder.build(directory, output, rl, 100_000, shard_bytes=100, stop_after_input_shards=1)
    staging = output.with_name(output.name + ".building")
    assert (staging / "checkpoint.json").exists()
    assert not (staging / "COMPLETE.json").exists()
    resumed = builder.build(directory, output, rl, 100_000, shard_bytes=100, resume=True)
    assert resumed == clean
    assert retained_rows(output, resumed) == retained_rows(uninterrupted, clean)


def test_batch_stop_resumes_source_row_and_removes_only_known_orphan(tmp_path, monkeypatch):
    directory, rl = checkpoint_fixture(tmp_path, monkeypatch,
                                      [source_row(f"Question {index}") for index in range(260)])
    calls = 0
    def stop():
        nonlocal calls
        calls += 1
        return calls >= 2
    output = tmp_path / "resumed"
    with pytest.raises(builder.PreparationInterrupted):
        builder.build(directory, output, rl, 10_000_000, stop_requested=stop)
    staging = output.with_name(output.name + ".building")
    saved = json.loads((staging / "checkpoint.json").read_text())
    assert saved["cursor"] == {"input_index": 0, "next_row": 128}
    orphan = staging / f"part-{len(saved['outputs']):05d}.parquet.tmp"
    orphan.write_bytes(b"interrupted parquet")
    resumed = builder.build(directory, output, rl, 10_000_000, resume=True)
    assert resumed["counts"] == {"examined": 260, "retained": 260}
    assert len(retained_rows(output, resumed)) == 260
    assert not (output / orphan.name).exists()


@pytest.mark.parametrize("tamper", ["output", "source", "checkpoint", "unknown_file", "target", "policy"])
def test_checkpoint_tampering_fails_closed(tmp_path, monkeypatch, tamper):
    directory, rl = checkpoint_fixture(tmp_path, monkeypatch)
    output = tmp_path / "resumed"
    with pytest.raises(builder.PreparationInterrupted):
        builder.build(directory, output, rl, 100_000, stop_after_input_shards=1)
    staging = output.with_name(output.name + ".building")
    if tamper == "output":
        path = next(staging.glob("*.parquet"))
        path.write_bytes(path.read_bytes() + b"changed")
    elif tamper == "source":
        path = directory / "part.parquet"
        path.write_bytes(path.read_bytes() + b"changed")
    elif tamper == "checkpoint":
        path = staging / "checkpoint.json"
        value = json.loads(path.read_text())
        value["counts"]["examined"] += 1
        path.write_text(json.dumps(value))
    elif tamper == "unknown_file":
        (staging / "valuable.txt").write_text("preserve me")
    elif tamper == "policy":
        changed = {"references": [], "rl_manifest_sha256": builder.common.sha256(rl), "new_policy": True}
        monkeypatch.setattr(builder.common, "load_references", lambda *_: (ProblemOverlapIndex([]), changed))
    with pytest.raises(ValueError):
        builder.build(directory, output, rl, 100_001 if tamper == "target" else 100_000, resume=True)
    assert not output.exists()
    if tamper == "unknown_file":
        assert (staging / "valuable.txt").read_text() == "preserve me"


def test_crash_after_output_before_checkpoint_replays_only_uncommitted_rows(tmp_path, monkeypatch):
    directory, rl = checkpoint_fixture(tmp_path, monkeypatch)
    output = tmp_path / "crash"
    write_json = builder.common.write_json
    calls = 0
    def crash(path, payload):
        nonlocal calls
        if path.name == "checkpoint.json":
            calls += 1
            if calls == 2:
                raise OSError("simulated failure before checkpoint replacement")
        write_json(path, payload)
    monkeypatch.setattr(builder.common, "write_json", crash)
    with pytest.raises(OSError, match="simulated"):
        builder.build(directory, output, rl, 100_000, shard_bytes=1)
    staging = output.with_name(output.name + ".building")
    assert (staging / "part-00000.parquet").exists()
    assert json.loads((staging / "checkpoint.json").read_text())["outputs"] == []
    monkeypatch.setattr(builder.common, "write_json", write_json)
    resumed = builder.build(directory, output, rl, 100_000, shard_bytes=1, resume=True)
    assert resumed["counts"] == {"examined": 3, "retained": 2, "duplicate_pair_within_or_against_base": 1}
    assert len(retained_rows(output, resumed)) == 2
