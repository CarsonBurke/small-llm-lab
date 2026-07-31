"""Download the pinned remote shards used by the K3-inspired pretraining mix.

This is a data-preparation workload. Run it through mlq:

    mlq submit --name k3_sources --cwd "$PWD" --max-parallel-runs 1 -- \
      python3 prepare_k3_pretrain_sources.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="pretraining/k3_sources.json")
    parser.add_argument("--output", default="data/pretraining_sources")
    parser.add_argument("--full-source-set", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, list[str]] = {}

    for source in manifest["sources"]:
        repo_id = source.get("repo_id")
        files = (
            source.get("full_files", source.get("files", []))
            if args.full_source_set
            else source.get("files", [])
        )
        if not repo_id:
            continue
        local_dir = output_root / source["name"]
        local_dir.mkdir(parents=True, exist_ok=True)
        resolved[source["name"]] = []
        for filename in files:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=source["revision"],
                repo_type="dataset",
                local_dir=local_dir,
            )
            resolved[source["name"]].append(str(Path(path).resolve()))

    provenance = {
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_version": manifest["version"],
        "full_source_set": args.full_source_set,
        "resolved_files": resolved,
    }
    output_path = output_root / "resolved_sources.json"
    output_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
