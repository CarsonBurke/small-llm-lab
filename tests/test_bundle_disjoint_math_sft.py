import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.bundle_disjoint_math_sft import bundle, sha256, write_json


def component(root, name, problem):
    directory = root / name
    directory.mkdir()
    rl = root / "rl.json"
    rl.write_text("{}")
    row = {"problem": problem, "generated_solution": "A complete solution"}
    path = directory / "part-00000.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)
    size = sum(len(value.encode()) for value in row.values())
    audit = {"rows": 1, "protected_question_matches": 0}
    manifest = {"source": {"repo_id": name}, "raw_text_bytes": size,
                "outputs": [{"path": path.name, "sha256": sha256(path),
                             "rows": 1, "raw_text_bytes": size, "bytes": path.stat().st_size}],
                "protection": {"references": [], "rl_manifest": str(rl),
                               "rl_manifest_sha256": sha256(rl)}, "audit": audit}
    write_json(directory / "audit.json", audit)
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "COMPLETE.json", {
        "manifest_sha256": sha256(directory / "manifest.json"),
        "audit_sha256": sha256(directory / "audit.json")})
    return directory / "manifest.json"


def test_bundle_counts_utf8_and_keeps_original_files(tmp_path):
    first = component(tmp_path, "first", "Compute π + 1")
    second = component(tmp_path, "second", "Compute 2 + 2")
    output = tmp_path / "bundle"
    result = bundle([first, second], output, 1)
    assert result["counts"]["retained"] == 2
    assert result["unique_questions"] == 2
    assert result["raw_text_bytes"] == sum(
        json.loads(path.read_text())["raw_text_bytes"] for path in [first, second])
    assert result["target_met"] is True
    assert not list(output.glob("*.parquet"))
    assert (output / "COMPLETE.json").exists()


def test_duplicate_pair_across_valid_components_is_rejected(tmp_path):
    first = component(tmp_path, "first", "Same question")
    second = component(tmp_path, "second", "Same question")
    with pytest.raises(ValueError, match="duplicate question/solution"):
        bundle([first, second], tmp_path / "bundle", 100)
    assert not (tmp_path / "bundle").exists()


def test_changed_shard_and_unlisted_shard_fail_closed(tmp_path):
    manifest = component(tmp_path, "first", "Question")
    extra = manifest.parent / "extra.parquet"
    extra.write_bytes(b"unlisted")
    with pytest.raises(ValueError, match="inventory"):
        bundle([manifest], tmp_path / "bundle", 100)
    extra.unlink()
    shard = manifest.parent / "part-00000.parquet"
    shard.write_bytes(shard.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="changed shard"):
        bundle([manifest], tmp_path / "bundle", 100)


def test_resume_incomplete_publication_and_completed_bundle(tmp_path):
    manifest = component(tmp_path, "first", "Question")
    output = tmp_path / "bundle"
    staging = tmp_path / "bundle.building"
    staging.mkdir()
    (staging / "audit.json").write_text('{"partial":')
    with pytest.raises(FileExistsError):
        bundle([manifest], output, 100)
    result = bundle([manifest], output, 100, resume=True)
    assert result == bundle([manifest], output, 100, resume=True)
    with pytest.raises(ValueError, match="resume parameters"):
        bundle([manifest], output, 101, resume=True)
    shard = manifest.parent / "part-00000.parquet"
    shard.write_bytes(shard.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="output changed"):
        bundle([manifest], output, 100, resume=True)


def test_resume_never_removes_unknown_staging_files(tmp_path):
    manifest = component(tmp_path, "first", "Question")
    staging = tmp_path / "bundle.building"
    staging.mkdir()
    unknown = staging / "unrelated.txt"
    unknown.write_text("keep")
    with pytest.raises(ValueError, match="unknown file"):
        bundle([manifest], tmp_path / "bundle", 100, resume=True)
    assert unknown.read_text() == "keep"
