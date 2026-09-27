"""Build immutable MBPP data and a VAPO mixture manifest.

A mixture is exact-pass: one cycle serves every row of every selected source
exactly once, interleaved in proportion to source size, so no prompt is
revisited before every other prompt has been seen. Sources therefore carry
no quota; composition is chosen by selecting (or rebuilding) sources.

``--sources`` selects a subset, because a mixture is only usable by a policy
whose SFT corpus taught every selected verifier's contract: pairing the code
source with a math-only SFT base yields rollouts that emit no think fence,
never terminate, and are format-ineligible for reward -- every code prompt
spent on guaranteed zeros, and the longest rollouts at that.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import warnings
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
from postraining.prepare_sft_corpus import refuse_evaluation_only
from postraining.vapo.code_reward import PYTHON_REWARD_SCHEMA, python_tests_pass
from postraining.vapo.mixture import VAPO_MIXTURE_SCHEMA, file_sha256


DEFAULT_OUTPUT = Path("postraining/data/vapo_broad_v12_exact")

# Each spec is (name, path, verifier, default-on). A frozen-policy gate on the
# post-SFT KDA8 policy (avg@512, 768-token budget) scored deepmind-interpolate
# 0.0000 and dapo-math-17k 0.0020; the v11 PG run later measured dapo at 0.004
# with within-group std 0.007. VAPO's advantage is group-relative, so an
# all-zero group contributes no policy gradient. ``deepmind_easy`` is the
# ``train-easy`` tier of the same 18 modules -- a disjoint training split at
# markedly easier surface difficulty -- and dominates the default mixture by
# size. The full-difficulty deepmind pool and MBPP are available but off by
# default: the former is unlearnable at the current accuracy, the latter
# needs a code-trained SFT base.
SOURCE_SPECS = (
    (
        "deepmind_easy",
        Path("postraining/data/deepmind-train-easy-rl.parquet"),
        "math",
        True,
    ),
    # v5: the whole UltraData-RL-2609 Math domain, decontaminated, within
    # the 256-token prompt budget, near-duplicates collapsed and equation
    # targets quarantined (scripts/build_ultradata_math_rl_prompts.py). It
    # owns every problem it shares with DAPO-Math-17K.
    (
        "ultradata_math",
        Path("postraining/data/ultradata-math-rl-v5.parquet"),
        "math",
        True,
    ),
    # v2: DAPO-Math-17K less every problem that restates an UltraData Math
    # problem (whitespace, LaTeX-skeleton, or near-duplicate match; 5,268
    # rows), with the same screens (scripts/build_dapo_rl_prompts.py).
    (
        "dapo",
        Path("postraining/data/dapo-math-17k-v2-ud3dedup.parquet"),
        "math",
        True,
    ),
    (
        "deepmind",
        Path("postraining/data/deepmind-interpolate-rl-full.parquet"),
        "math",
        False,
    ),
    ("mbpp", None, "python_mbpp", False),
    # openbmb/UltraData-Code L3/py exercises verified by
    # ``prepare_ultradata_code build``: the prompt shows at most two example
    # assertions and the reward runs every kept test statement. Off by
    # default for the same reason as mbpp -- it needs a code-SFT'd base --
    # and its SFT counterpart is the disjoint ``ultradata_code_l3`` pool.
    (
        "ultradata_code_l3",
        Path("postraining/data/ultradata-code-l3-v3-rl.parquet"),
        "python_mbpp",
        False,
    ),
    # UltraData-RL-2609 Knowledge (STEM), built by
    # ``scripts/build_ultradata_knowledge_rl_prompts.py``: single-choice rows
    # re-rendered as "A. option" lines with the answer permuted to a uniform
    # position and graded as one exact letter, plus numeric rows. Off by
    # default: guessing earns 1/k per row (the ``choice_{k}`` module baselines
    # report it), so it needs a gate showing accuracy above chance first, and
    # a base whose SFT taught letter answers (``ultradata_sft_2605_knowledge``).
    (
        "ultradata_knowledge",
        Path("postraining/data/ultradata-knowledge-rl-v2.parquet"),
        "math",
        False,
    ),
    # ARC-Challenge, ARC-Easy, OpenBookQA and SciQ train splits, built by
    # ``scripts/build_science_mc_rl_prompts.py`` under the same single-choice
    # contract (``{source}_choice_{k}`` modules, chance mostly 1/4). Off by
    # default for the same reason as ``ultradata_knowledge``; SciQ rows are
    # CC BY-NC 3.0. v5 is v2 less the question components the teacher-trace
    # SFT source (``science_mc_traces``) distils, so SFT and RL share no
    # question (v3 and v4 split reworded questions apart; neither was used);
    # the job 9040 base scored 0.2% on v2 (job 9466), so it needs a base whose
    # SFT included that source.
    (
        "science_mc",
        Path("postraining/data/science-mc-rl-v5.parquet"),
        "math",
        False,
    ),
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


def mbpp_entry_points(reference: str, test_sources: list[str]) -> list[str]:
    """Reference definitions the fixture and tests read.

    The v6 verifier shows the tests only these candidate bindings: names the
    reference binds at module scope (imports excluded, since tests import
    for themselves) that the test code reads but never binds.
    """

    def module_scope(tree: ast.Module, *, imports: bool) -> set[str]:
        names: set[str] = set()
        for statement in tree.body:
            if isinstance(
                statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                names.add(statement.name)
            elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                if imports:
                    names.update(
                        (alias.asname or alias.name).split(".", 1)[0]
                        for alias in statement.names
                    )
            else:
                names.update(
                    node.id
                    for node in ast.walk(statement)
                    if isinstance(node, ast.Name)
                    and isinstance(node.ctx, (ast.Store, ast.Del))
                )
        return names

    def parse(source: str) -> ast.Module:
        # MBPP sources carry non-raw regex escapes; the sandbox reports them.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(source)

    defined = module_scope(parse(reference), imports=False)
    read: set[str] = set()
    bound: set[str] = set()
    for source in test_sources:
        tree = parse(source)
        read.update(
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        )
        bound |= module_scope(tree, imports=True)
    entry_points = sorted((defined & read) - bound)
    if not entry_points:
        raise ValueError("MBPP tests read no reference definition")
    return entry_points


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
                    "entry_points": mbpp_entry_points(
                        str(source["code"]),
                        ([setup] if setup else []) + visible_tests,
                    ),
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
            name for name, _, _, default_on in SOURCE_SPECS if default_on
        ),
        help="comma-separated subset of "
        f"{','.join(name for name, _, _, _ in SOURCE_SPECS)}; every row of "
        "each selected source is served exactly once per mixture cycle",
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
    specs = [
        (name, path, verifier)
        for name, path, verifier, _ in SOURCE_SPECS
        if name in set(selected)
    ]
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
    for name, configured_path, verifier in specs:
        path = mbpp_path if configured_path is None else configured_path
        try:
            refuse_evaluation_only(path)
        except ValueError as error:
            parser.error(str(error))
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
                "verifier": verifier,
                "rows": len(rows),
                "sha256": file_sha256(path),
                "math_corpus_identity": math_corpus_identity(rows),
                "math_corpus_audit": corpus_audit,
            }
        )
    # A prompt reachable through two sources is drawn under two names, counted
    # twice against the pool, and splits its own learnability telemetry. The
    # pools must be made disjoint at build time (postraining/problem_overlap.py,
    # scripts/build_dapo_rl_prompts.py) rather than reconciled here,
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
        "prompts_per_cycle": sum(source["rows"] for source in sources),
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
