#!/usr/bin/env python3
"""Download pinned math SFT sources; HF_TOKEN stays in the environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

SOURCES = {
    "openmathreasoning": ("nvidia/OpenMathReasoning",
                          "d3d08664755704f422af97d43a7ff0ded4bd95df",
                          "data/cot-", 144, "cc-by-4.0"),
    "nemotron-v4": ("nvidia/Nemotron-SFT-Math-v4",
                    "84d42ad0cb960f07f951b9baa9ed2b46a5a18c66",
                    "data/train-", 12, ["cc-by-4.0", "cc-by-sa-4.0"]),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--source", choices=sorted(SOURCES), default="openmathreasoning")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    token = os.environ.get("HF_TOKEN")
    if not token:
        parser.error("provide HF_TOKEN in the environment, never as a command argument")
    api = HfApi(token=token)
    api.whoami()
    repo, revision, prefix, shard_count, license_name = SOURCES[args.source]
    info = api.dataset_info(repo, revision=revision, files_metadata=True)
    files = [{"path": item.rfilename, "bytes": item.size,
              "sha256": item.lfs.sha256 if item.lfs else None}
             for item in sorted(info.siblings, key=lambda entry: entry.rfilename)
             if item.rfilename.startswith(prefix)]
    if len(files) != shard_count or any(not item["sha256"] for item in files):
        raise ValueError("pinned source has an unexpected shard inventory")
    manifest = {"repo_id": repo, "revision": revision, "license": license_name, "files": files}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "source_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("destination contains a different source manifest")
    else:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(path)
    print(json.dumps({"phase": "download", "shards": len(files),
                      "bytes": sum(item["bytes"] for item in files),
                      "revision": revision}), flush=True)
    snapshot_download(repo, repo_type="dataset", revision=revision,
                      allow_patterns=[prefix + "*", "README.md"],
                      local_dir=args.output_dir, max_workers=args.workers, token=token)
    for item in files:
        if (args.output_dir / item["path"]).stat().st_size != item["bytes"]:
            raise ValueError(f"download size differs: {item['path']}")
    print("Download complete; corpus preparation verifies every consumed SHA-256.", flush=True)


if __name__ == "__main__":
    main()
