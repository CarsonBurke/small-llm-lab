"""Read-only RL corpus census and sandboxed trivial-program audit; run via mlq."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re

import pyarrow.parquet as pq

from postraining.core import GPT2BPETokenizer, load_unique_math_rows
from postraining.math_prompt import answer_fence_prompt
from postraining.prepare_vapo_mixture import SOURCE_SPECS, atomic_json
from postraining.vapo.code_reward import python_test_result
from postraining.vapo.mixture import file_sha256


def literal_truth(node):
    """Only fold provably constant assertions; never execute corpus code."""
    if isinstance(node, ast.IfExp):
        condition = literal_truth(node.test)
        if condition is not None:
            return literal_truth(node.body if condition else node.orelse)
    if isinstance(node, ast.BoolOp):
        values = [literal_truth(value) for value in node.values]
        if isinstance(node.op, ast.Or) and True in values:
            return True
        if isinstance(node.op, ast.And) and False in values:
            return False
    try:
        return bool(ast.literal_eval(node))
    except (ValueError, TypeError):
        return None


def audit_mutants(row):
    info = row["verification_info"]
    outcomes = {}
    witnesses = []
    # No access to hidden expected answers when constructing these candidates.
    for expression in ("None", "0", "1", "True", "False", "[]", "{}", "''", "args[0]", "args[-1]"):
        candidate = "\n".join(
            f"def {name}(*args, **kwargs):\n    return {expression}\n"
            for name in info["entry_points"]
        )
        result = python_test_result(candidate, info)
        outcomes[expression] = result
        if result == "pass":
            witnesses.append(candidate)
    return {"id": row["extra_info"]["index"], "outcomes": outcomes,
            "passing_candidates": witnesses, "prompt": row["prompt"][0]["content"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mutant-prompts", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite audit evidence")
    tokenizer = GPT2BPETokenizer()
    report = {"schema": "rl_corpus_quality/v1", "sources": {}}
    code_rows = None
    for name, path, _, default_on in SOURCE_SPECS:
        if not default_on and name not in {"ultradata_code_l3", "science_mc", "ultradata_knowledge"}:
            continue
        rows = load_unique_math_rows(path)
        lengths = []
        modules = Counter()
        keys = Counter()
        labels = {}
        overlong = []
        identities = Counter()
        for row in rows:
            info = row.get("extra_info") or {}
            prompt = row["prompt"][0]["content"]
            # Math pools retain source framing; the trainer removes it before
            # encoding. Audit the served problem, including its leading BOS.
            length = 1 + len(tokenizer.encode(answer_fence_prompt(prompt)))
            lengths.append(length)
            if length > 256:
                overlong.append(info.get("index"))
            module = str(info.get("module", name))
            modules[module] += 1
            keys.update(info.keys())
            identities[re.sub(r"\s+", " ", prompt).strip().lower()] += 1
            if "choice_" in module:
                labels.setdefault(module, Counter())[row["reward_model"]["ground_truth"]] += 1
        ordered = sorted(lengths)
        entry = {
            "path": str(path), "sha256": file_sha256(path),
            "physical_rows": pq.read_metadata(path).num_rows, "effective_rows": len(rows),
            "modules": dict(modules), "metadata_keys": dict(keys),
            "choice_label_counts": {key: dict(value) for key, value in labels.items()},
            "whitespace_case_duplicate_rows": sum(n - 1 for n in identities.values()),
            "prompt_tokens_including_bos": {"max": max(lengths), "p50": ordered[len(ordered)//2],
                                           "p95": ordered[int(.95 * (len(ordered)-1))]},
            "over_256_token_ids": overlong,
        }
        if name == "ultradata_code_l3":
            code_rows = rows
            vacuous = []
            for row in rows:
                for statement in row["verification_info"]["tests"]:
                    tree = ast.parse(statement)
                    if any(isinstance(node, ast.Assert) and literal_truth(node.test) is True
                           for node in ast.walk(tree)):
                        vacuous.append({"id": row["extra_info"]["index"], "statement": statement})
            entry["vacuous_assertions"] = vacuous
            entry["vacuous_assertion_rows"] = len({item["id"] for item in vacuous})
        report["sources"][name] = entry
        print(json.dumps({"source": name, "rows": len(rows), "overlong": len(overlong)}), flush=True)
    assert code_rows is not None
    code_rows.sort(key=lambda row: hashlib.sha256(
        ("quality-v1" + row["extra_info"]["index"]).encode()).hexdigest())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        mutations = list(pool.map(audit_mutants, code_rows[:args.mutant_prompts]))
    report["code_mutations"] = {
        "selection": "sha256(quality-v1 + extra_info.index)",
        "prompts": len(mutations), "trivial_program_pass_rows": sum(bool(r["passing_candidates"]) for r in mutations),
        "results": mutations,
        "interpretation": "A passing trivial program needs semantic adjudication; identity/constant functions can be legitimate tasks.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(report, args.output)
    print(json.dumps({"output": str(args.output), "trivial_program_pass_rows": report["code_mutations"]["trivial_program_pass_rows"]}))


if __name__ == "__main__":
    main()
