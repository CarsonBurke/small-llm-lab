"""The held-out drill prompts the start policy rarely solves: a hard panel.

Selects the bench prompts whose contract-correct count at a captured bench
step lies in [--min-correct, --max-correct] of the capture's samples
(default 1-4 of 16: p in (0, 0.25]). These are the prompts where an
answer-level reallocation toward hard prompts (DG's Prop. 2) should show:
the policy can find the answer but rarely does. 0/16 prompts carry no
learnable group signal for any group estimator and are excluded.

Writes, refusing to overwrite:
- ``<output>.parquet``: the selected rows of ``--bench-data``, in its row
  order, byte-identical in schema, usable as ``--bench-data`` with
  ``--bench-max-rows 0``;
- ``<output>.manifest.json``: the selection rule, SHA-256 of every input,
  and per prompt its bench ``problem_index`` (its position in the training
  run's bench panel, which indexes ``contract_prompt_correct_counts``), its
  dataset index and its step-0 count.

The selection reuses the start policy's own draws, so the panel's step-0
mean is biased low (regression to the mean). DG and its RLOO control share
the start policy and the panel, so their paired difference is not biased
by it; an arm's gain over step 0 on this panel is.

    python3 scripts/build_hard_bench_panel.py \\
        --bench-data postraining/data/drill_rl_study_heldout_20261014_v1.parquet \\
        --answers postraining/runs/drill_step_error_bench_s000_20261021/bench_answers/step_000000.json \\
        --output postraining/data/drill_rl_study_heldout_hard_s000_20261021_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

PANEL_SCHEMA = "drill_bench_hard_panel_step_correct_count_band/v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bench-data", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True,
                        help="bench_answers/step_*.json capturing every problem")
    parser.add_argument("--min-correct", type=int, default=1)
    parser.add_argument("--max-correct", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True,
                        help="output path without suffix")
    args = parser.parse_args()
    parquet_path = args.output.with_suffix(".parquet")
    manifest_path = args.output.with_suffix(".manifest.json")
    for path in (parquet_path, manifest_path):
        if path.exists():
            raise SystemExit(f"{path} exists; panels are immutable, version a new one")
    capture = json.loads(args.answers.read_text())
    counts = capture["metrics"]["contract_prompt_correct_counts"]
    samples = capture["samples_per_problem"]
    if capture["problem_count"] != len(counts):
        raise SystemExit("the capture must cover every bench problem")
    if not 0 <= args.min_correct <= args.max_correct <= samples:
        raise SystemExit("need 0 <= --min-correct <= --max-correct <= samples")
    # problem_index -> dataset index, from the capture's own attempts.
    dataset_index: dict[int, str] = {}
    per_problem: dict[int, int] = {}
    for attempt in capture["attempts"]:
        problem = int(attempt["problem_index"])
        if dataset_index.setdefault(problem, attempt["dataset_index"]) != attempt["dataset_index"]:
            raise SystemExit(f"problem {problem} maps to two dataset rows")
        per_problem[problem] = per_problem.get(problem, 0) + 1
    if sorted(dataset_index) != list(range(len(counts))) or set(per_problem.values()) != {samples}:
        raise SystemExit("the capture must hold every sample of every problem")
    # The capture's counts must be its attempts' own.
    for problem, count in enumerate(counts):
        captured = sum(
            bool(attempt["contract_correct"]) for attempt in capture["attempts"]
            if int(attempt["problem_index"]) == problem)
        if captured != count:
            raise SystemExit(f"problem {problem}: count {count} but {captured} captured")
    selected = {
        dataset_index[problem]: problem for problem, count in enumerate(counts)
        if args.min_correct <= count <= args.max_correct
    }
    table = pq.read_table(args.bench_data)
    indices = [info["index"] for info in table.column("extra_info").to_pylist()]
    if len(set(indices)) != len(indices):
        raise SystemExit("bench rows must have unique extra_info.index")
    keep = [row for row, index in enumerate(indices) if index in selected]
    if len(keep) != len(selected):
        raise SystemExit("a selected problem is missing from --bench-data")
    panel = table.take(keep)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(panel, parquet_path)
    rows = [
        {"dataset_index": indices[row], "problem_index": selected[indices[row]],
         "step_correct": counts[selected[indices[row]]],
         "module": table.column("extra_info")[row].as_py().get("module")}
        for row in keep
    ]
    manifest = {
        "schema": PANEL_SCHEMA,
        "selection": {
            "rule": "contract_prompt_correct_counts in [min_correct, max_correct]",
            "min_correct": args.min_correct, "max_correct": args.max_correct,
            "samples": samples, "capture_step": capture["step"],
        },
        "inputs": {
            "bench_data": str(args.bench_data), "bench_data_sha256": file_sha256(args.bench_data),
            "answers": str(args.answers), "answers_sha256": file_sha256(args.answers),
        },
        "output_sha256": file_sha256(parquet_path),
        "bench_problems": len(counts),
        "rows": rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    modules: dict[str, int] = {}
    for row in rows:
        modules[row["module"]] = modules.get(row["module"], 0) + 1
    print(json.dumps({"selected": len(rows), "of": len(counts), "modules": modules}, indent=2))


if __name__ == "__main__":
    main()
