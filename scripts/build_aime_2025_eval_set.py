"""Acquire the 30 held-out AIME 2025 problems; no model or GPU executes.

Uses pinned OpenCompass text transcriptions, retaining upstream source bytes and
provenance. Reuses the AIME 2026 builder's DAPO-compatible schema and writer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow.parquet as pq
from scripts.build_aime_2026_eval_set import (
    atomic_write_bytes, atomic_write_parquet, convert, download,
)

SOURCE_REPOSITORY = "opencompass/AIME2025"
SOURCE_REVISION = "a6ad95f611d72cf628a80b58bd0432ef6638f958"
SOURCE_HASHES = {
    "I": "b91b3c96f05d9635d2a0692b124ebe023c1ff59cb19c074275e6c4b349d0659e",
    "II": "16a2dcfbbf9db1b11f8a69a3ba5e4cac73e3641b19a37e2307e9c12240bbed5e",
}
# Cross-checked against math-ai/aime25 at
# 563bb8404243c5f09de6ec262f2db674fe5bce9b (same order and integer answers).
EXPECTED_ANSWERS = (
    70, 588, 16, 117, 279, 504, 821, 77, 62, 81, 259, 510, 204, 60, 735,
    468, 49, 82, 106, 336, 293, 237, 610, 149, 907, 113, 19, 248, 104, 240,
)


def parse_form(payload: bytes, form: str) -> list[dict]:
    if hashlib.sha256(payload).hexdigest() != SOURCE_HASHES[form]:
        raise ValueError(f"AIME 2025 {form} source hash differs from the pinned bytes")
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if len(rows) != 15:
        raise ValueError(f"AIME 2025 {form} must contain 15 problems")
    offset = 0 if form == "I" else 15
    converted = []
    for number, row in enumerate(rows, start=1):
        problem = row["question"]
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError("source problem is empty")
        if any(ord(character) < 32 and character not in "\n\t\r" for character in problem):
            raise ValueError("source problem contains an escaped LaTeX control character")
        answer = int(str(row["answer"]).removesuffix(r"^\circ"))
        source_id = offset + number
        if answer != EXPECTED_ANSWERS[source_id - 1]:
            raise ValueError(f"AIME 2025 {form} #{number} answer differs from the pinned key")
        converted.append({"id": source_id, "problem": problem, "answer": answer})
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("postraining/data"))
    args = parser.parse_args()
    source_rows = []
    sources = []
    payloads = []
    for form in ("I", "II"):
        url = f"https://huggingface.co/datasets/{SOURCE_REPOSITORY}/resolve/{SOURCE_REVISION}/aime2025-{form}.jsonl"
        payload = download(url)
        source_rows.extend(parse_form(payload, form))
        raw_file = f"aime-2025-{form.lower()}-source.jsonl"
        payloads.append((raw_file, payload))
        sources.append({"form": form, "url": url, "sha256": SOURCE_HASHES[form], "raw_file": raw_file})
    rows = convert(source_rows, year=2025)
    if len({row["extra_info"]["raw_problem"] for row in rows}) != 30:
        raise ValueError("AIME 2025 contains duplicate problem statements")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for raw_file, payload in payloads:
        atomic_write_bytes(args.output_dir / raw_file, payload)
    outputs = {}
    for suffix, selected in (
        ("", rows),
        ("-i", [row for row in rows if row["extra_info"]["form"] == "I"]),
        ("-ii", [row for row in rows if row["extra_info"]["form"] == "II"]),
    ):
        path = args.output_dir / f"aime-2025{suffix}.parquet"
        atomic_write_parquet(path, selected)
        if pq.read_table(path).to_pylist() != selected:
            raise AssertionError(f"persisted dataset differs: {path}")
        outputs[path.name] = {"problems": len(selected), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "sources": sources,
        "outputs": outputs,
        "problems": 30,
        "form_i_problems": 15,
        "form_ii_problems": 15,
        "template": "verbatim DAPO-Math-17K prompt wrapper",
        "upstream_declared_license": "MIT",
        "copyright_note": "Contest questions are MAA-copyrighted; resolve redistribution rights before publishing local files.",
        "answer_normalization": "AIME II #5: remove degree suffix from 336^\\circ; integer value unchanged",
        "answer_key_crosscheck": "math-ai/aime25@563bb8404243c5f09de6ec262f2db674fe5bce9b",
        "publisher_dataset_match_verified": False,
        "evaluation_limitations": [
            "Uses OpenCompass text transcriptions; diagrams are omitted or represented as text.",
            "The publisher's exact dataset revision and prompt formatting remain unverified.",
        ],
    }
    atomic_write_bytes(args.output_dir / "aime-2025.manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
