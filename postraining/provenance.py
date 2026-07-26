"""Reproducible source snapshots for long-running post-training jobs."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import torch

_SOURCE_SUFFIXES = {
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_EXCLUDED_TOP_LEVEL = {
    ".git",
    "ablation_results",
    "data",
    "postraining/data",
    "postraining/runs",
    "runs",
    "tb_logs",
}


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ("git", *args),
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout


def _source_paths(
    repo: Path, *, excluded_root: Path | None = None
) -> list[Path]:
    listed = _git(
        repo,
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
    )
    paths = []
    for raw in listed.split(b"\0"):
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        relative_text = relative.as_posix()
        if excluded_root is not None and (
            relative == excluded_root
            or excluded_root in relative.parents
        ):
            continue
        if any(
            relative_text == prefix
            or relative_text.startswith(prefix + "/")
            for prefix in _EXCLUDED_TOP_LEVEL
        ):
            continue
        path = repo / relative
        if path.is_file() and path.suffix.lower() in _SOURCE_SUFFIXES:
            paths.append(relative)
    return sorted(paths, key=lambda path: path.as_posix())


def capture_source_provenance(output: Path, repo: Path) -> dict[str, object]:
    """Atomically archive the actual tracked and untracked source checkout."""
    try:
        excluded_root = output.parent.resolve().relative_to(repo.resolve())
        if excluded_root == Path("."):
            excluded_root = None
    except ValueError:
        excluded_root = None
    status_lines = _git(repo, "status", "--short").decode(
        errors="surrogateescape"
    ).splitlines()
    if excluded_root is not None:
        excluded_prefix = excluded_root.as_posix()
        status_lines = [
            line
            for line in status_lines
            if not (
                line[3:] == excluded_prefix
                or line[3:].startswith(excluded_prefix + "/")
            )
        ]
    paths = _source_paths(repo, excluded_root=excluded_root)
    git_head = _git(repo, "rev-parse", "HEAD").decode().strip()
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / f".source.snapshot.{os.getpid()}.tmp"
    with temporary.open("wb") as raw_archive:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_archive, mtime=0
        ) as compressed:
            with tarfile.open(
                fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
            ) as tar:
                for relative in paths:
                    path = repo / relative
                    info = tar.gettarinfo(
                        path, arcname=relative.as_posix()
                    )
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    if info.isfile():
                        with path.open("rb") as source:
                            tar.addfile(info, source)
                    else:
                        tar.addfile(info)
    archive_sha256 = hashlib.sha256(temporary.read_bytes()).hexdigest()
    archive = output / (
        f"source.{git_head[:12]}.{archive_sha256[:12]}.snapshot.tar.gz"
    )
    if archive.exists():
        temporary.unlink()
    else:
        os.replace(temporary, archive)
    metadata: dict[str, object] = {
        "git_head": git_head,
        "git_status": status_lines,
        "source_archive": str(archive.relative_to(output.parent)),
        "source_archive_sha256": archive_sha256,
        "source_file_count": len(paths),
        "argv": list(sys.argv),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
    }
    metadata_text = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    metadata_path = output / "metadata.json"
    temporary_metadata = output / f".metadata.{os.getpid()}.tmp"
    temporary_metadata.write_text(metadata_text)
    os.replace(temporary_metadata, metadata_path)
    metadata_sha256 = hashlib.sha256(metadata_text.encode()).hexdigest()
    immutable_metadata = output / (
        f"metadata.{archive_sha256[:12]}.{metadata_sha256[:12]}.json"
    )
    if not immutable_metadata.exists():
        temporary_immutable = output / (
            f".metadata.history.{os.getpid()}.tmp"
        )
        temporary_immutable.write_text(metadata_text)
        os.replace(temporary_immutable, immutable_metadata)
    return metadata
