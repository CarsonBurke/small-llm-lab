"""Fetch the relaxed-bar SFT trace sources to local parquets.

Downloads the corpus blend chosen from the 2026-08-02 relaxed-bar survey
(NOTES.md) into ``postraining/data/relaxed_bar/``. Fetch-time filtering
is limited to cheap ROW SELECTION that avoids storing bulk we will never
use (OpenMathInstruct-2's augmented_math majority — ~22% ill-posed with
self-consistent boxed answers — is excluded here by problem_source);
every semantic repair, verification, and decontamination step happens in
prepare_sft_traces where it is tested.

Each source skips if its output parquet already exists — delete a file
to refetch it.

    python3 -m postraining.fetch_relaxed_bar_sources
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

OUT_DIR = Path("postraining/data/relaxed_bar")

# problem_source values worth keeping from OpenMathInstruct-2: the two
# GSM8K-band subsets with 0/830-scale measured defect rates. The
# augmented_math 83% majority is excluded (survey: ~22% ill-posed, and
# expected_answer matches the boxed value by construction, so no
# downstream check can catch the bad rows).
OPENMATH_SOURCES = ("augmented_gsm8k", "gsm8k")


def write_parquet(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError(f"no rows fetched for {path}")
    table = pa.Table.from_pylist(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    print(f"wrote {len(rows)} rows -> {path}", flush=True)


def fetch_openmathinstruct2(path: Path) -> None:
    # Streamed: only the ~153k kept rows ever materialize; the split is
    # 1M rows and the excluded majority never touches disk.
    stream = load_dataset(
        "nvidia/OpenMathInstruct-2", split="train_1M", streaming=True
    )
    rows = []
    started = time.perf_counter()
    for index, row in enumerate(stream):
        if row["problem_source"] in OPENMATH_SOURCES:
            rows.append(
                {
                    "problem": row["problem"],
                    "generated_solution": row["generated_solution"],
                    "expected_answer": row["expected_answer"],
                    "problem_source": row["problem_source"],
                }
            )
        if (index + 1) % 100_000 == 0:
            print(
                f"openmathinstruct2: {index + 1} scanned, {len(rows)} kept, "
                f"{time.perf_counter() - started:.0f}s",
                flush=True,
            )
    write_parquet(rows, path)


def fetch_a1_math_deepmind(path: Path) -> None:
    data = load_dataset("mlfoundations-dev/a1_math_deepmind", split="train")
    keep = (
        "question",
        "answer",
        "deepseek_solution",
        "instruction_seed",
        "source",
    )
    rows = [
        {key: row.get(key) for key in keep}
        for row in data
    ]
    write_parquet(rows, path)


def fetch_had653(path: Path) -> None:
    data = load_dataset(
        "HAD653/GSM8K-OpenMath-MathReason-13k", split="train"
    )
    rows = [
        {
            "question": row["question"],
            "cot": row["cot"],
            "final_answer": row["final_answer"],
        }
        for row in data
    ]
    write_parquet(rows, path)


def fetch_sxiong(path: Path) -> None:
    data = load_dataset("sxiong/synthetic-math", split="train")
    keep = ("id", "problem", "solution", "answer", "type", "level")
    rows = [{key: row.get(key) for key in keep} for row in data]
    write_parquet(rows, path)


def fetch_gsm8k_socratic(path: Path) -> None:
    data = load_dataset("openai/gsm8k", "socratic", split="train")
    rows = [
        {"question": row["question"], "answer": row["answer"]} for row in data
    ]
    write_parquet(rows, path)


def fetch_gsm8k_main_train(path: Path) -> None:
    # Gold for the HAD653 join (57% of its questions are verbatim GSM8K
    # train; only those rows carry independent ground truth).
    data = load_dataset("openai/gsm8k", "main", split="train")
    rows = [
        {"question": row["question"], "answer": row["answer"]} for row in data
    ]
    write_parquet(rows, path)


def fetch_gsm8k_test(path: Path) -> None:
    # Decontamination target only — never a training source.
    data = load_dataset("openai/gsm8k", "main", split="test")
    rows = [{"question": row["question"]} for row in data]
    write_parquet(rows, path)


FETCHERS = {
    "openmathinstruct2_gsm8k_band.parquet": fetch_openmathinstruct2,
    "a1_math_deepmind.parquet": fetch_a1_math_deepmind,
    "had653_gsm8k_openmath.parquet": fetch_had653,
    "sxiong_synthetic_math.parquet": fetch_sxiong,
    "gsm8k_socratic_train.parquet": fetch_gsm8k_socratic,
    "gsm8k_main_train.parquet": fetch_gsm8k_main_train,
    "gsm8k_test_questions.parquet": fetch_gsm8k_test,
}


def main() -> None:
    manifest: dict[str, int] = {}
    for name, fetch in FETCHERS.items():
        path = OUT_DIR / name
        if path.exists():
            print(f"skip {name} (exists)", flush=True)
        else:
            print(f"fetching {name}...", flush=True)
            fetch(path)
        manifest[name] = pq.read_metadata(path).num_rows
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {json.dumps(manifest, indent=2)}")


if __name__ == "__main__":
    main()
