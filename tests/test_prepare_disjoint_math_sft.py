from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import prepare_disjoint_math_sft as builder
from postraining.problem_overlap import ProblemOverlapIndex


def row(problem="Find the value of 17 + 29.", solution=r"Adding gives \boxed{46}"):
    return {"problem": problem, "generated_solution": solution, "expected_answer": "46",
            "inference_mode": "cot", "problem_type": "has_answer_extracted",
            "generation_model": "teacher", "problem_source": "MATH_training_set"}


def source(tmp_path, rows):
    directory = tmp_path / "source"
    directory.mkdir()
    path = directory / "cot.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    builder.write_json(directory / "source_manifest.json", {
        "repo_id": "nvidia/OpenMathReasoning", "revision": "fixture-pinned", "license": "cc-by-4.0",
        "files": [{"path": path.name, "sha256": builder.sha256(path), "bytes": path.stat().st_size}],
    })
    return directory


def test_balanced_terminal_box_and_conservative_answer_agreement():
    assert builder.terminal_box(r"Thus \boxed{\frac{1}{2}}\]") == r"\frac{1}{2}"
    assert builder.terminal_box(r"Thus \boxed{2} but maybe not") is None
    assert builder.terminal_box(r"Thus \boxed{\frac{1}{2}") is None
    assert builder.answer_key(r"\( \dfrac{1}{2} \)") == builder.answer_key(r"\frac{1}{2}")
    assert builder.answer_key(r"\\text{No}") == builder.answer_key(r"\text{No}")
    assert builder.answer_key("1+2") != builder.answer_key("1-2")


def test_answer_normalization_respects_latex_command_boundaries_and_matrix_rows():
    assert builder.answer_key(r"\leftarrow") != builder.answer_key(r"\rightarrow")
    assert builder.answer_key(r"\left( \dfrac{1}{2} \right)") == builder.answer_key(r"(\frac{1}{2})")
    assert builder.answer_key(r"\leftarrow") == r"\leftarrow"
    assert builder.answer_key(r"\dfracfoo") == r"\dfracfoo"
    matrix = r"\begin{matrix}1\\x\end{matrix}"
    different = r"\begin{matrix}1\x\end{matrix}"
    assert builder.answer_key(matrix) == matrix
    assert builder.answer_key(matrix) != builder.answer_key(different)
    assert builder.answer_key(matrix.replace("\\", "\\\\")) != builder.answer_key(matrix)


def test_normalization_preserves_content_and_rejects_nested_fences():
    original = row(solution="<think>Reasoning\n</think>Then \\boxed{46}")
    normalized = builder.normalize_row(original)
    assert normalized["generated_solution"] == "Reasoning\nThen \\boxed{46}"
    assert original["generated_solution"].startswith("<think>")
    assert builder.quality_reason(normalized) is None
    assert builder.quality_reason(builder.normalize_row(row(solution="<think><think>x</think></think>\\boxed{46}"))) == "reserved_fence_or_tool_artifact"
    assert builder.quality_reason(row(solution="<tool_call>print(46)</tool_call>\\boxed{46}")) == "reserved_fence_or_tool_artifact"
    assert builder.quality_reason({**row(), "expected_answer": "47"}) == "boxed_expected_text_disagreement"
    assert builder.quality_reason({**row(), "inference_mode": "tir"}) == "non_cot"


def test_inputs_fail_closed_for_missing_extra_or_changed_shard(tmp_path):
    directory = source(tmp_path, [row()])
    records, _ = builder.validate_inputs(directory)
    assert records[0]["rows"] == 1
    path = directory / "cot.parquet"
    original = path.read_bytes()
    path.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="hash/size"):
        builder.validate_inputs(directory)
    path.write_bytes(original)
    (directory / "extra.parquet").write_bytes(original)
    with pytest.raises(ValueError, match="shard set"):
        builder.validate_inputs(directory)


def test_reference_stems_screen_options_variants():
    refs = builder.reference_questions([{"question": "What is 7+3?\n\nA. 10\nB. 11"}])
    assert "What is 7+3?" in refs


def test_candidate_framing_cannot_hide_short_protected_question():
    problem = "What is 7+3?"
    candidate = builder.normalize_row(row("Solve the following math problem step by step.\n\n" + problem))
    assert candidate["problem"] == problem
    assert ProblemOverlapIndex([problem]).matches(candidate["problem"])[0].matcher == "whitespace"


def test_pending_inputs_validate_manifest_then_fail_closed_if_not_arrived(tmp_path):
    directory = source(tmp_path, [row()])
    (directory / "cot.parquet").unlink()
    records, _ = builder.validate_inputs(directory, allow_pending=True)
    with pytest.raises(FileNotFoundError):
        builder.verify_input(directory, records[0], deadline=0)
    with pytest.raises(ValueError, match="shard set"):
        builder.validate_inputs(directory)


def test_build_deduplicates_screens_and_independently_audits(tmp_path, monkeypatch):
    rows = [row(), row(), row("Protected problem"), row("A distinct question", "<think>Work</think>\\boxed{46}")]
    directory = source(tmp_path, rows)
    index = ProblemOverlapIndex(["Protected problem"])
    monkeypatch.setattr(builder, "load_references", lambda *_: (index, {"references": []}))
    output = tmp_path / "output"
    manifest = builder.build(directory, output, tmp_path / "unused", 10_000, (), shard_bytes=25)
    assert manifest["counts"]["retained"] == 2
    assert manifest["counts"]["duplicate_question_solution"] == 1
    assert manifest["counts"]["protected_whitespace"] == 1
    assert manifest["target_met"] is False
    assert manifest["audit"]["protected_question_matches"] == 0
    assert (output / "COMPLETE.json").exists()
    retained = pq.read_table(output / manifest["outputs"][1]["path"]).to_pylist()[0]
    assert retained["source_row"] == 3
    assert retained["generated_solution"] == "Work\\boxed{46}"
    with pytest.raises(FileExistsError):
        builder.build(directory, output, tmp_path / "unused", 1, ())
    path = output / manifest["outputs"][0]["path"]
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="Output hash"):
        builder.audit_output(output, index, manifest["outputs"])


def test_independent_audit_rejects_contamination_even_with_correct_hash(tmp_path):
    path = tmp_path / "part.parquet"
    pq.write_table(pa.Table.from_pylist([row("Protected problem")]), path)
    with pytest.raises(ValueError, match="overlaps"):
        builder.audit_output(tmp_path, ProblemOverlapIndex(["Protected problem"]),
                             [{"path": path.name, "sha256": builder.sha256(path), "rows": 1}])


def test_stem_target_declaration_includes_prerevision_and_science():
    targets = builder.stem_targets()
    assert any(column == "Pre-Revision Question" for _, column in targets)
    assert any("ARC-Challenge/test" in str(path) for path, _ in targets)


def checkpoint_source(tmp_path, monkeypatch, shards=None):
    directory = tmp_path / "source"
    directory.mkdir()
    shards = shards or [
        [row("A"), row("A"), row("Protected problem"), {**row("bad"), "expected_answer": "47"}],
        [row("A"), row("B"), {**row("tir"), "inference_mode": "tir"}, row("C")],
    ]
    records = []
    for number, rows in enumerate(shards):
        path = directory / f"input-{number:03d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        records.append({"path": path.name, "bytes": path.stat().st_size, "sha256": builder.sha256(path)})
    builder.write_json(directory / "source_manifest.json", {
        "repo_id": "nvidia/OpenMathReasoning", "revision": "fixture-pinned", "files": records,
    })
    monkeypatch.setattr(builder, "load_references", lambda *_: (ProblemOverlapIndex(["Protected problem"]), {"references": []}))
    return directory


def logical_rows(output, manifest):
    return [row for item in manifest["outputs"] for row in pq.read_table(output / item["path"]).to_pylist()]


def test_checkpoint_pause_resume_matches_uninterrupted_rows_and_counters(tmp_path, monkeypatch):
    source_dir = checkpoint_source(tmp_path, monkeypatch)
    output = tmp_path / "resumed"
    paused = builder.build(source_dir, output, tmp_path / "rl", 100_000, (), stop_after_input_shards=1)
    stage = tmp_path / "resumed.building"
    assert paused["paused"] is True
    assert not output.exists() and not (stage / "COMPLETE.json").exists()
    saved = builder._read_checkpoint(stage)
    assert (saved["next_input"], saved["next_row"]) == (1, 0)
    assert saved["counts"]["examined"] == 4
    resumed = builder.build(source_dir, output, tmp_path / "rl", 100_000, (), resume=True)
    fresh_output = tmp_path / "fresh"
    fresh = builder.build(source_dir, fresh_output, tmp_path / "rl", 100_000, ())
    assert logical_rows(output, resumed) == logical_rows(fresh_output, fresh)
    for key in ("counts", "raw_text_bytes", "unique_questions", "character_length_histogram", "solutions_per_question_histogram"):
        assert resumed[key] == fresh[key]


@pytest.mark.parametrize("tamper,match", [
    ("output", "Committed output hash"), ("source", "Input hash/size"),
    ("target", "binding mismatch"), ("shard_size", "binding mismatch"),
    ("checkpoint", "checksum"), ("source_manifest", "binding mismatch"),
    ("unknown_file", "Unknown staging file"),
])
def test_resume_refuses_tampering_and_configuration_changes(tmp_path, monkeypatch, tamper, match):
    source_dir = checkpoint_source(tmp_path, monkeypatch)
    output = tmp_path / "output"
    builder.build(source_dir, output, tmp_path / "rl", 100_000, (), stop_after_input_shards=1)
    stage = tmp_path / "output.building"
    target, shard_size = 100_000, 512_000_000
    if tamper == "output":
        path = stage / "part-00000.parquet"
        path.write_bytes(path.read_bytes() + b"changed")
    elif tamper == "source":
        path = source_dir / "input-000.parquet"
        path.write_bytes(path.read_bytes() + b"changed")
    elif tamper == "checkpoint":
        path = stage / "checkpoint.json"
        content = json.loads(path.read_text())
        content["payload"]["counts"]["examined"] += 1
        path.write_text(json.dumps(content))
    elif tamper == "source_manifest":
        path = source_dir / "source_manifest.json"
        content = json.loads(path.read_text())
        content["revision"] = "different"
        path.write_text(json.dumps(content))
    elif tamper == "unknown_file":
        (stage / "user_notes.txt").write_text("preserve me")
    elif tamper == "target":
        target += 1
    else:
        shard_size += 1
    with pytest.raises(ValueError, match=match):
        builder.build(source_dir, output, tmp_path / "rl", target, (), shard_bytes=shard_size, resume=True)
    if tamper == "unknown_file":
        assert (stage / "user_notes.txt").read_text() == "preserve me"


def test_resume_discards_only_recognized_uncommitted_trailing_artifacts(tmp_path, monkeypatch):
    source_dir = checkpoint_source(tmp_path, monkeypatch)
    output = tmp_path / "output"
    builder.build(source_dir, output, tmp_path / "rl", 100_000, (), stop_after_input_shards=1)
    stage = tmp_path / "output.building"
    for name in ("part-00001.parquet", "part-00001.parquet.tmp", "checkpoint.json.tmp"):
        (stage / name).write_bytes(b"interrupted write")
    resumed = builder.build(source_dir, output, tmp_path / "rl", 100_000, (), resume=True)
    assert resumed["counts"]["retained"] == 3
    assert not list(output.glob("*.tmp"))


def test_forced_interruption_replays_from_last_committed_output(tmp_path, monkeypatch):
    source_dir = checkpoint_source(tmp_path, monkeypatch, [[row("A"), row("B"), row("A"), row("C")]])
    output = tmp_path / "output"
    original_write = builder.write_json
    writes = 0

    def crash_checkpoint(path, value):
        nonlocal writes
        if path.name == "checkpoint.json":
            writes += 1
            if writes == 3:
                raise RuntimeError("simulated forced interruption")
        original_write(path, value)

    monkeypatch.setattr(builder, "write_json", crash_checkpoint)
    with pytest.raises(RuntimeError, match="forced interruption"):
        builder.build(source_dir, output, tmp_path / "rl", 100_000, (), shard_bytes=1)
    saved = builder._read_checkpoint(tmp_path / "output.building")
    assert saved["next_row"] == 1
    assert (tmp_path / "output.building/part-00001.parquet").exists()
    monkeypatch.setattr(builder, "write_json", original_write)
    resumed = builder.build(source_dir, output, tmp_path / "rl", 100_000, (), shard_bytes=1, resume=True)
    assert resumed["counts"] == {"examined": 4, "retained": 3, "duplicate_question_solution": 1}
    assert [r["problem"] for r in logical_rows(output, resumed)] == ["A", "B", "C"]


def test_sigterm_cli_pauses_at_batch_boundary_with_exit_75(tmp_path, monkeypatch):
    source_dir = checkpoint_source(tmp_path, monkeypatch, [[row(str(number)) for number in range(140)]])
    output = tmp_path / "output"
    original_normalize = builder.normalize_row
    sent = False

    def request_signal(row):
        nonlocal sent
        if not sent:
            sent = True
            os.kill(os.getpid(), signal.SIGTERM)
        return original_normalize(row)

    monkeypatch.setattr(builder, "normalize_row", request_signal)
    monkeypatch.setattr(sys, "argv", ["prepare_disjoint_math_sft.py", "--input-dir", str(source_dir),
                                      "--output-dir", str(output), "--target-bytes", "100000"])
    assert builder.main() == 75
    assert builder._read_checkpoint(tmp_path / "output.building")["counts"]["examined"] == 128
    assert not (tmp_path / "output.building/COMPLETE.json").exists()


def test_pause_during_final_audit_keeps_complete_data_checkpoint(tmp_path, monkeypatch):
    source_dir = checkpoint_source(tmp_path, monkeypatch)
    output = tmp_path / "output"
    original_audit = builder.audit_output

    def interrupt_audit(*args, **kwargs):
        raise builder.PreparationPaused

    monkeypatch.setattr(builder, "audit_output", interrupt_audit)
    assert builder.build(source_dir, output, tmp_path / "rl", 100_000, ())["paused"]
    stage = tmp_path / "output.building"
    assert builder._read_checkpoint(stage)["data_complete"] is True
    for name in ("audit.json", "manifest.json.tmp", "COMPLETE.json"):
        (stage / name).write_text("interrupted final metadata")
    monkeypatch.setattr(builder, "audit_output", original_audit)
    resumed = builder.build(source_dir, output, tmp_path / "rl", 100_000, (), resume=True)
    assert resumed["counts"]["retained"] == 3
    assert resumed["audit"]["protected_question_matches"] == 0
