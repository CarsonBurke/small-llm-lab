"""Build immutable MBPP data and a VAPO mixture manifest.

The default composition is the four-source math+code mixture. ``--sources``
selects a subset, because a mixture is only usable by a policy whose SFT
corpus taught every selected verifier's contract: pairing the code source
with a math-only SFT base yields rollouts that emit no think fence, never
terminate, and are format-ineligible for reward -- a whole quota of every
pool spent on guaranteed zeros, and the longest rollouts at that.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from postraining.core import (
    load_unique_math_rows,
    math_corpus_identity,
    math_corpus_policy_sha256,
)
from postraining.math_prompt import (
    ANSWER_FENCE_PROMPT_SCHEMA,
    answer_fence_prompt,
)
from postraining.vapo.code_reward import PYTHON_REWARD_SCHEMA, python_tests_pass
from postraining.vapo.mixture import VAPO_MIXTURE_SCHEMA, file_sha256


DEFAULT_OUTPUT = Path("postraining/data/vapo_broad_v9_bare")

# Quotas follow measured learnability, not source size.  A frozen-policy gate
# on the post-SFT KDA8 policy (avg@512, 768-token budget) scored:
#
#   deepmind-interpolate (train pool and bench panel)   0.0000
#   dapo-math-17k                                       0.0020
#
# VAPO's advantage is group-relative, so an all-zero group contributes exactly
# no policy gradient -- weighting the hardest sources most, as the v7 mixture
# did (dapo 28/64, deepmind 20/64), spent 87% of every rollout pool on prompts
# that cannot teach anything.  ``deepmind_easy`` is the ``train-easy`` tier of
# the same 18 modules: a disjoint training split at markedly easier surface
# difficulty, which is where a 0.00 policy can first produce within-group
# variance.  The harder tail is kept deliberately small -- enough to preserve
# headroom and to stop the policy narrowing onto one templated prompt shape,
# which is itself a documented failure mode -- and should be reweighted upward
# as train-easy accuracy climbs.
SOURCE_SPECS = (
    (
        "deepmind_easy",
        Path("postraining/data/deepmind-train-easy-rl.parquet"),
        48,
        "math",
    ),
    # v2 of this pool: dapo-overlapping prompts removed. UltraData-RL-2609
    # and DAPO-Math-17K share upstream pools, and 664 prompts were reachable
    # through both.
    (
        "ultradata_math",
        Path("postraining/data/ultradata-math-rl-v2.parquet"),
        8,
        "math",
    ),
    (
        "dapo",
        Path("postraining/data/dapo-math-17k.parquet"),
        8,
        "math",
    ),
    (
        "deepmind",
        Path("postraining/data/deepmind-interpolate-rl-full.parquet"),
        0,
        "math",
    ),
    ("mbpp", None, 0, "python_mbpp"),
)

# Selecting one of these is a mistake with a specific explanation, so name the
# reason instead of reporting it as an unknown source.
RETIRED_SOURCES = {
    "gsm8k": (
        "gsm8k is an evaluation set for this project; training RL on it would "
        "make every reported gsm8k number a training-set score"
    ),
}


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
    parser.add_argument(
        "--mbpp-jsonl",
        help="required only when the mbpp source is selected",
    )
    parser.add_argument("--output-prefix", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--sft-corpus",
        required=True,
        help="SFT corpus the policy was distilled on; bound into the manifest "
        "by sha256 so RL cannot resume against a different base",
    )
    parser.add_argument(
        "--sources",
        default=",".join(
            name for name, _, quota, _ in SOURCE_SPECS if quota
        ),
        help="comma-separated subset of "
        f"{','.join(name for name, _, _, _ in SOURCE_SPECS)}; quotas are the "
        "shipped per-source values, so a subset changes groups_per_cycle. "
        "Sources shipped at quota 0 are available but off by default and "
        "must be given an explicit --quota override to contribute",
    )
    parser.add_argument(
        "--quota",
        action="append",
        default=[],
        metavar="SOURCE=N",
        help="override a source's prompts-per-pool quota; repeatable",
    )
    args = parser.parse_args()

    known = {name for name, _, _, _ in SOURCE_SPECS}
    selected = [name for name in args.sources.split(",") if name]
    for name in selected:
        if name in RETIRED_SOURCES:
            parser.error(f"source {name!r} is retired: {RETIRED_SOURCES[name]}")
    unknown = sorted(set(selected) - known)
    if unknown:
        parser.error(f"unknown --sources entries: {', '.join(unknown)}")
    if not selected:
        parser.error("--sources selects no source")
    if len(selected) != len(set(selected)):
        parser.error("--sources repeats a source")
    overrides = {}
    for item in args.quota:
        name, _, raw = item.partition("=")
        if name not in known:
            parser.error(f"--quota names unknown source {name!r}")
        if name not in set(selected):
            parser.error(f"--quota {name!r} is not in --sources")
        if not raw.isdigit() or int(raw) < 1:
            parser.error(f"--quota {item!r} needs a positive integer")
        overrides[name] = int(raw)
    specs = [
        (name, path, overrides.get(name, quota), reward)
        for name, path, quota, reward in SOURCE_SPECS
        if name in set(selected)
    ]
    zero = sorted(name for name, _, quota, _ in specs if quota == 0)
    if zero:
        parser.error(
            f"selected source(s) {', '.join(zero)} ship quota 0; give an "
            "explicit --quota SOURCE=N to include them"
        )
    wants_mbpp = any(spec[0] == "mbpp" for spec in specs)
    if wants_mbpp and not args.mbpp_jsonl:
        parser.error("--mbpp-jsonl is required when the mbpp source is selected")
    if args.mbpp_jsonl and not wants_mbpp:
        parser.error("--mbpp-jsonl given but the mbpp source is not selected")

    prefix = Path(args.output_prefix)
    mbpp_path = prefix.with_name(prefix.name + "_mbpp_train.parquet")
    mbpp_source_path = prefix.with_name(prefix.name + "_mbpp_source.jsonl")
    manifest_path = prefix.with_suffix(".manifest.json")
    outputs = [manifest_path]
    if wants_mbpp:
        outputs += [mbpp_path, mbpp_source_path]
    for path in outputs:
        if path.exists():
            parser.error(f"refusing to overwrite immutable output {path}")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    if wants_mbpp:
        raw_mbpp = Path(args.mbpp_jsonl)
        mbpp_rows = load_mbpp_train(raw_mbpp)
        atomic_bytes(raw_mbpp.read_bytes(), mbpp_source_path)
        atomic_parquet(mbpp_rows, mbpp_path)

    sources = []
    prompt_owners: dict[str, list[str]] = {}
    for name, configured_path, quota, verifier in specs:
        path = mbpp_path if configured_path is None else configured_path
        corpus_audit = {}
        rows = load_unique_math_rows(path, audit=corpus_audit)
        for row in rows:
            try:
                canonical = answer_fence_prompt(row["prompt"][0]["content"])
            except (ValueError, KeyError, IndexError, TypeError):
                continue
            prompt_owners.setdefault(
                re.sub(r"\s+", " ", canonical).strip().lower(), []
            ).append(name)
        sources.append(
            {
                "name": name,
                "path": str(path),
                "quota": quota,
                "verifier": verifier,
                "rows": len(rows),
                "sha256": file_sha256(path),
                "math_corpus_identity": math_corpus_identity(rows),
                "math_corpus_audit": corpus_audit,
            }
        )
    # A prompt reachable through two sources is drawn under two names, counted
    # twice against the pool, and splits its own learnability telemetry. The
    # pools must be made disjoint at build time (see the --exclude option on
    # scripts/build_ultradata_math_rl_prompts.py) rather than reconciled here,
    # because dropping rows now would silently change a source's row count and
    # its math_corpus_identity.
    shared = {
        prompt: owners
        for prompt, owners in prompt_owners.items()
        if len(set(owners)) > 1
    }
    if shared:
        pairs = Counter(
            tuple(sorted(set(owners))) for owners in shared.values()
        )
        detail = "; ".join(
            f"{' + '.join(pair)}: {count}" for pair, count in pairs.most_common()
        )
        example = next(iter(shared))[:120]
        parser.error(
            f"{len(shared)} prompt(s) appear in more than one selected "
            f"source ({detail}). Rebuild the overlapping pool with the other "
            f"excluded. First: {example!r}"
        )

    sft_path = Path(args.sft_corpus)
    manifest = {
        "schema": VAPO_MIXTURE_SCHEMA,
        "math_corpus_policy_sha256": math_corpus_policy_sha256(),
        "groups_per_cycle": sum(source["quota"] for source in sources),
        "sources": sources,
        "answer_fence_prompt_schema": ANSWER_FENCE_PROMPT_SCHEMA,
        "python_reward_schema": PYTHON_REWARD_SCHEMA,
        "sft_corpus": str(sft_path),
        "sft_corpus_sha256": file_sha256(sft_path),
    }
    if wants_mbpp:
        manifest["mbpp_source"] = {
            "path": str(mbpp_source_path),
            "sha256": file_sha256(mbpp_source_path),
            "official_url": (
                "https://raw.githubusercontent.com/google-research/"
                "google-research/master/mbpp/mbpp.jsonl"
            ),
            "split": "task_id_601_974",
        }
    atomic_json(manifest, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
