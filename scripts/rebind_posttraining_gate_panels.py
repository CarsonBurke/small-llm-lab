"""Bind unchanged diagnostic panels to another SFT lineage; run through mlq."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-panel-dir", required=True, type=Path)
    parser.add_argument("--sft-corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sources", nargs="+", default=["deepmind_easy", "ultradata_math", "dapo"])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(set(args.sources)) != len(args.sources):
        raise ValueError("duplicate sources")
    corpus_hash = sha256(args.sft_corpus)
    manifests = []
    for source in args.sources:
        if Path(source).name != source:
            raise ValueError("source must be a single file-name component")
        template = args.template_panel_dir / f"{source}.manifest.json"
        manifest = json.loads(template.read_text())
        for entry in manifest["sources"]:
            if sha256(Path(entry["path"])) != entry["sha256"]:
                raise ValueError(f"panel hash changed: {entry['path']}")
        manifest["sft_corpus"] = str(args.sft_corpus)
        manifest["sft_corpus_sha256"] = corpus_hash
        manifest["lineage_rebinding"] = {
            "template_manifest": str(template),
            "template_manifest_sha256": sha256(template),
            "scope": "SFT binding only; identical panel bytes, order, source names, and verifier contracts",
            "interpretation": "RL-pool diagnostic; cross-lineage training overlap is not excluded",
        }
        manifests.append((source, manifest))
    args.output.mkdir(parents=True, exist_ok=False)
    for source, manifest in manifests:
        with (args.output / f"{source}.manifest.json").open("x") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps({"output": str(args.output), "sources": args.sources, "sft_corpus_sha256": corpus_hash}))


if __name__ == "__main__":
    main()
