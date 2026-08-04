"""Build immutable MBPP data and the four-source VAPO mixture manifest."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from postraining.core import load_unique_math_rows
from postraining.math_prompt import ANSWER_FENCE_PROMPT_SCHEMA
from postraining.vapo.code_reward import PYTHON_REWARD_SCHEMA, python_tests_pass
from postraining.vapo.mixture import VAPO_MIXTURE_SCHEMA, file_sha256


DEFAULT_OUTPUT = Path("postraining/data/vapo_broad_v5")
DEFAULT_SFT = Path(
    "postraining/data/sft_traces_v4_answer_canonical_hfonly.parquet"
)
SOURCE_SPECS = (
    (
        "dapo",
        Path("postraining/data/dapo-math-17k.parquet"),
        28,
        "math",
    ),
    (
        "deepmind",
        Path("postraining/data/deepmind-interpolate-rl-full.parquet"),
        20,
        "math",
    ),
    (
        "gsm8k",
        Path("postraining/data/gsm8k_rl_prompts.parquet"),
        8,
        "math",
    ),
    ("mbpp", None, 8, "python_mbpp"),
)


def atomic_parquet(rows: list[dict], path: Path) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    os.replace(temporary, path)


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_bytes(payload: bytes, path: Path) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def load_mbpp_train(path: Path) -> list[dict]:
    source_rows = [json.loads(line) for line in path.read_text().splitlines()]
    selected = [row for row in source_rows if 601 <= int(row["task_id"]) <= 974]
    if len(selected) != 374:
        raise ValueError(f"expected 374 MBPP train tasks, found {len(selected)}")
    rows = []
    for source in selected:
        visible_tests = [str(test) for test in source["test_list"]]
        setup = str(source.get("test_setup_code") or "").strip()
        fixture = (setup + "\n") if setup else ""
        prompt = (
            f"{str(source['text']).strip()}\n\n"
            "Your final answer must be a complete executable Python module. "
            "Use deterministic in-process Python only: no filesystem, process, "
            "network, reflection, dynamic execution, or interactive I/O. "
            "Use only ordinary task data fields and collection/string/math "
            "methods; interpreter and frame attributes are unavailable. "
            "Imports are limited to bisect, cmath, collections, datetime, "
            "heapq, itertools, math, re, sys.maxsize, and operator.eq. "
            "It must define the requested function and pass this fixture and "
            "all tests:\n"
            + fixture
            + "\n".join(visible_tests)
        )
        rows.append(
            {
                "data_source": "mbpp_train",
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "coding",
                # The generic VAPO loader requires this field, but the Python
                # verifier ignores it and executes verification_info instead.
                "reward_model": {
                    "ground_truth": "ALL_TESTS_PASS",
                    "style": "python-exec",
                },
                "extra_info": {
                    "index": f"mbpp_train_{int(source['task_id'])}",
                    "module": "mbpp",
                    "prompt_contract": "bare",
                },
                "verification_info": {
                    "schema": PYTHON_REWARD_SCHEMA,
                    "test_setup": [setup] if setup else [],
                    "tests": visible_tests,
                },
                # Removed after a mandatory verifier preflight and never
                # written into the RL parquet.
                "_reference_solution": str(source["code"]),
            }
        )
    def reference_passes(row: dict) -> bool:
        return python_tests_pass(
            row["_reference_solution"], row["verification_info"]
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        passed = list(pool.map(reference_passes, rows))
    failed = [
        row["extra_info"]["index"]
        for row, success in zip(rows, passed, strict=True)
        if not success
    ]
    if failed:
        raise ValueError(
            f"MBPP verifier rejected official reference solutions: {failed}"
        )
    for row in rows:
        del row["_reference_solution"]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mbpp-jsonl", required=True)
    parser.add_argument("--output-prefix", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sft-corpus", default=str(DEFAULT_SFT))
    args = parser.parse_args()

    prefix = Path(args.output_prefix)
    mbpp_path = prefix.with_name(prefix.name + "_mbpp_train.parquet")
    mbpp_source_path = prefix.with_name(prefix.name + "_mbpp_source.jsonl")
    manifest_path = prefix.with_suffix(".manifest.json")
    for path in (mbpp_path, mbpp_source_path, manifest_path):
        if path.exists():
            parser.error(f"refusing to overwrite immutable output {path}")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    raw_mbpp = Path(args.mbpp_jsonl)
    mbpp_rows = load_mbpp_train(raw_mbpp)
    atomic_bytes(raw_mbpp.read_bytes(), mbpp_source_path)
    atomic_parquet(mbpp_rows, mbpp_path)

    sources = []
    for name, configured_path, quota, verifier in SOURCE_SPECS:
        path = mbpp_path if configured_path is None else configured_path
        rows = load_unique_math_rows(path)
        sources.append(
            {
                "name": name,
                "path": str(path),
                "quota": quota,
                "verifier": verifier,
                "rows": len(rows),
                "sha256": file_sha256(path),
            }
        )
    sft_path = Path(args.sft_corpus)
    manifest = {
        "schema": VAPO_MIXTURE_SCHEMA,
        "groups_per_cycle": sum(source["quota"] for source in sources),
        "sources": sources,
        "answer_fence_prompt_schema": ANSWER_FENCE_PROMPT_SCHEMA,
        "python_reward_schema": PYTHON_REWARD_SCHEMA,
        "sft_corpus": str(sft_path),
        "sft_corpus_sha256": file_sha256(sft_path),
        "mbpp_source": {
            "path": str(mbpp_source_path),
            "sha256": file_sha256(mbpp_source_path),
            "official_url": (
                "https://raw.githubusercontent.com/google-research/"
                "google-research/master/mbpp/mbpp.jsonl"
            ),
            "split": "task_id_601_974",
        },
    }
    atomic_json(manifest, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
