"""Sample a completed math SFT corpus for context coverage, without a model.

Run CPU preprocessing through mlq. This audit never truncates examples and
reports sampled estimates, not exact corpus-wide token statistics.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import pyarrow.parquet as pq
from transformers import GPT2TokenizerFast


SPECIALS = ["<think>", "</think>", "<answer>", "</answer>"]
WINDOWS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
REQUIRED = ("problem", "generated_solution", "expected_answer")
OPTIONAL = ("problem_source", "source", "source_repo", "source_revision",
            "source_shard", "source_row", "source_row_index", "generation_model",
            "expected_answer_source", "inference_mode", "final_answer", "problem_type")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_shards(manifest: dict, manifest_path: Path) -> list[tuple[Path, dict]]:
    entries = manifest.get("outputs", manifest.get("output_shards", manifest.get("shards")))
    if entries is None and isinstance(manifest.get("output"), dict):
        entries = manifest["output"].get("shards")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest must list nonempty outputs, output_shards or shards")
    shards = []
    for entry in entries:
        metadata = {"path": entry} if isinstance(entry, str) else entry
        path = Path(metadata.get("path", metadata.get("file", "")))
        if not path.is_absolute():
            candidates = [manifest_path.parent / path, Path.cwd() / path]
            existing = {candidate.resolve() for candidate in candidates if candidate.is_file()}
            if len(existing) != 1:
                raise ValueError(f"missing or ambiguous shard path: {path}")
            path = existing.pop()
        if not path.is_file():
            raise FileNotFoundError(path)
        shards.append((path, metadata))
    if len({path for path, _ in shards}) != len(shards):
        raise ValueError("manifest lists a shard more than once")
    return shards


def reservoir(shards: list[tuple[Path, dict]], size: int, seed: int):
    rng = random.Random(seed)
    sample = []
    seen = 0
    coverage = []
    for path, metadata in shards:
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        if not set(REQUIRED).issubset(names):
            raise ValueError(f"required columns missing in {path}")
        columns = list(REQUIRED) + [name for name in OPTIONAL if name in names]
        offset = 0
        for batch in parquet.iter_batches(batch_size=512, columns=columns):
            # Algorithm R; decode Python strings only for selected records.
            selected = {}
            for position in range(batch.num_rows):
                slot = seen if seen < size else rng.randrange(seen + 1)
                seen += 1
                if slot < size:
                    selected[slot] = position
            for slot, position in sorted(selected.items()):
                row = {name: batch.column(index)[position].as_py()
                       for index, name in enumerate(columns)}
                row["audit_output_shard"] = str(path)
                row["audit_output_row"] = offset + position
                if metadata.get("component_manifest") is not None:
                    row["component_manifest"] = metadata["component_manifest"]
                if slot < len(sample):
                    sample[slot] = row
                else:
                    sample.append(row)
            offset += batch.num_rows
        expected = metadata.get("rows", metadata.get("num_rows"))
        if expected is not None and offset != expected:
            raise ValueError(f"row count differs from manifest: {path}")
        shard_coverage = {"path": str(path), "rows": offset,
                          "bytes": path.stat().st_size,
                          "manifest_sha256": metadata.get("sha256")}
        if metadata.get("component_manifest") is not None:
            shard_coverage["component_manifest"] = metadata["component_manifest"]
        coverage.append(shard_coverage)
        print(json.dumps({"phase": "sample", "shards_scanned": len(coverage),
                          "rows_scanned": seen}), flush=True)
    if not sample:
        raise ValueError("empty corpus")
    return sample, seen, coverage


def quantiles(values: list[int]) -> dict:
    return {str(q): float(np.quantile(values, q))
            for q in (0, .1, .25, .5, .75, .9, .95, .99, 1)}


def audit(sample: list[dict]) -> dict:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
    tokenizer.model_max_length = 1 << 30
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIALS})
    special_ids = tokenizer.convert_tokens_to_ids(SPECIALS)
    if special_ids != [50257, 50258, 50259, 50260]:
        raise ValueError(f"special IDs differ from core tokenizer: {special_ids}")
    lengths, prompts, completions, text_bytes, records = [], [], [], [], []
    for row in sample:
        if any(not isinstance(row[key], str) or not row[key].strip() for key in REQUIRED):
            raise ValueError("sample has an empty or nonstring required field")
        if any(token in row[key] for key in REQUIRED for token in SPECIALS):
            raise ValueError("sample contains reserved fence tokens")
        problem, solution, expected = (row[key] for key in REQUIRED)
        answer = row.get("final_answer", expected)
        if not isinstance(answer, str) or not answer.strip() or any(token in answer for token in SPECIALS):
            raise ValueError("sample has an invalid final_answer")
        completion = f"<think>\n{solution}\n</think>\n<answer>{answer}</answer>"
        # Matches sft_trace_train.tokenize_documents: encode separately,
        # prefix BOS, then reserve a trailing stop/separator target.
        prompt_tokens = 1 + len(tokenizer.encode(problem, add_special_tokens=False))
        completion_tokens = len(tokenizer.encode(completion, add_special_tokens=False))
        length = prompt_tokens + completion_tokens + 1
        lengths.append(length)
        raw_bytes = len(problem.encode("utf-8")) + len(solution.encode("utf-8"))
        text_bytes.append(raw_bytes)
        prompts.append(prompt_tokens)
        completions.append(completion_tokens)
        records.append({**{key: value for key, value in row.items()
                           if key not in (*REQUIRED, "final_answer")},
                        "problem_sha256": hashlib.sha256(problem.encode()).hexdigest(),
                        "prompt_tokens_with_bos": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "raw_text_bytes": raw_bytes,
                        "trained_tokens_with_stop": length})
    return {
        "tokenizer": {"name": "gpt2", "local_files_only": True,
                      "special_tokens": dict(zip(SPECIALS, special_ids)),
                      "backend_sha256": hashlib.sha256(
                          tokenizer.backend_tokenizer.to_str().encode()).hexdigest()},
        "framing": "bare problem, BOS + separately encoded prompt/completion + stop; final_answer when present, otherwise expected_answer; no truncation",
        "fit": {str(window): {"rows": sum(value <= window for value in lengths),
                              "fraction": sum(value <= window for value in lengths) / len(lengths),
                              "sample_raw_text_bytes_fitting": sum(
                                  size for length, size in zip(lengths, text_bytes) if length <= window),
                              "estimated_raw_text_byte_fraction": sum(
                                  size for length, size in zip(lengths, text_bytes) if length <= window
                              ) / sum(text_bytes)}
                for window in WINDOWS},
        "sample_raw_text_bytes": sum(text_bytes),
        "raw_text_byte_fit_method": "Ratio of fitting sampled problem-plus-generated_solution UTF-8 bytes to all sampled bytes; a ratio estimate from a uniform row sample, not exact corpus volume. Excludes answer framing and metadata, matching corpus size definition.",
        "quantiles": {"trained_tokens_with_stop": quantiles(lengths),
                      "prompt_tokens_with_bos": quantiles(prompts),
                      "completion_tokens": quantiles(completions)},
        "sample_source_counts": dict(Counter(str(row.get("problem_source", row.get("source", "unknown")))
                                              for row in sample)),
        "sample": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse a completed matching audit, or restart the sample scan from saved corpus shards")
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")
    manifest_digest = sha256(args.manifest)
    script_digest = sha256(Path(__file__))
    if args.output.is_symlink() or (args.output.exists() and not args.output.is_file()):
        raise ValueError(f"report must be a regular file: {args.output}")
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        previous = json.loads(args.output.read_text())
        if (previous.get("schema") != "disjoint_math_sft_context_audit/v1"
                or previous.get("manifest_sha256") != manifest_digest
                or previous.get("auditor_sha256") != script_digest
                or previous.get("sampling", {}).get("seed") != args.seed
                or previous.get("sampling", {}).get("requested_rows") != args.sample_size):
            raise ValueError("existing audit does not match resume inputs/configuration")
        print(json.dumps({"resumed_completed_audit": str(args.output),
                          "fit": previous["fit"]}), flush=True)
        return
    temporary = args.output.with_name(args.output.name + ".tmp")
    if temporary.is_symlink() or (temporary.exists() and not temporary.is_file()):
        raise ValueError(f"temporary report must be a regular file: {temporary}")
    if temporary.exists() and not args.resume:
        raise FileExistsError(f"interrupted report write: use --resume ({temporary})")
    manifest = json.loads(args.manifest.read_text())
    shards = manifest_shards(manifest, args.manifest)
    sample, total, coverage = reservoir(shards, args.sample_size, args.seed)
    expected_rows = manifest.get("counts", {}).get("retained")
    if expected_rows is not None and total != expected_rows:
        raise ValueError("corpus row total differs from manifest")
    result = {"schema": "disjoint_math_sft_context_audit/v1",
              "auditor_sha256": script_digest,
              "manifest": str(args.manifest.resolve()), "manifest_sha256": manifest_digest,
              "source_provenance": {key: manifest.get("source", {}).get(key)
                                    for key in ("repo_id", "revision", "manifest_sha256")},
              "component_provenance": [
                  {"manifest": component["manifest"],
                   "manifest_sha256": component["sha256"],
                   "audit_sha256": component.get("audit_sha256"),
                   "source": {key: component.get("source", {}).get(key)
                              for key in ("repo_id", "revision", "manifest_sha256")},
                   "rows": component.get("rows"),
                   "raw_text_bytes": component.get("raw_text_bytes")}
                  for component in manifest.get("components", [])],
              "sampling": {"method": "Algorithm R uniform reservoir over every output row",
                           "seed": args.seed, "requested_rows": args.sample_size,
                           "sampled_rows": len(sample), "corpus_rows": total,
                           "fraction": len(sample) / total},
              "coverage": coverage, **audit(sample),
              "limitations": ["Token statistics and fit fractions are sample estimates.",
                              "No model ran; token fit does not establish training effectiveness.",
                              "Shard hashes are recorded from the manifest, not reverified by this length audit."]}
    if sha256(args.manifest) != manifest_digest:
        raise ValueError("manifest changed while auditing")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(args.output)
    print(json.dumps({key: result[key] for key in ("sampling", "fit", "quantiles")}), flush=True)


if __name__ == "__main__":
    main()
