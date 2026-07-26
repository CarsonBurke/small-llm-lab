from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tarfile

from postraining.provenance import capture_source_provenance


def _git(repo: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=repo, check=True, capture_output=True)


def test_source_provenance_captures_tracked_dirty_and_untracked_source(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "tracked.py").write_text("before = 1\n")
    (repo / "weights.pt").write_bytes(b"not source")
    _git(repo, "add", "tracked.py", "weights.pt")
    _git(repo, "commit", "-m", "fixture")
    (repo / "tracked.py").write_text("after = 2\n")
    (repo / "untracked.py").write_text("new = 3\n")
    (repo / "data").mkdir()
    (repo / "data" / "ignored.py").write_text("large = True\n")
    monkeypatch.setattr(
        "sys.argv", ["trainer.py", "--output", "run"]
    )

    run = repo / "out"
    metadata = capture_source_provenance(run / "provenance", repo)

    archives = list(
        (run / "provenance").glob("source.*.snapshot.tar.gz")
    )
    assert len(archives) == 1
    with tarfile.open(archives[0], "r:gz") as archive:
        names = set(archive.getnames())
        assert archive.extractfile("tracked.py").read() == b"after = 2\n"
        assert archive.extractfile("untracked.py").read() == b"new = 3\n"
    assert names == {"tracked.py", "untracked.py"}
    assert metadata["source_file_count"] == 2
    assert " M tracked.py" in metadata["git_status"]
    assert "?? untracked.py" in metadata["git_status"]
    assert metadata["argv"] == ["trainer.py", "--output", "run"]
    assert len(metadata["source_archive_sha256"]) == 64
    assert json.loads(
        (run / "provenance" / "metadata.json").read_text()
    ) == metadata
    assert len(
        list((run / "provenance").glob("metadata.*.*.json"))
    ) == 1

    first_archive = archives[0]
    first_history = next(
        (run / "provenance").glob("metadata.*.*.json")
    )
    unchanged = capture_source_provenance(run / "provenance", repo)
    assert unchanged["source_archive"] == metadata["source_archive"]
    assert unchanged["source_archive_sha256"] == (
        metadata["source_archive_sha256"]
    )
    assert len(
        list((run / "provenance").glob("source.*.snapshot.tar.gz"))
    ) == 1

    (repo / "tracked.py").write_text("resumed = 4\n")
    resumed = capture_source_provenance(run / "provenance", repo)

    assert first_archive.exists()
    assert first_history.exists()
    assert len(
        list((run / "provenance").glob("source.*.snapshot.tar.gz"))
    ) == 2
    assert len(
        list((run / "provenance").glob("metadata.*.*.json"))
    ) == 2
    assert resumed["source_archive"] != metadata["source_archive"]
    assert json.loads(
        (run / "provenance" / "metadata.json").read_text()
    ) == resumed
