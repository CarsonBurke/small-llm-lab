"""Pinned KodCode adaptation and metrics for the existing sandboxed code reward.

This is deliberately a restricted executable subset, not a pytest replacement.
No dataset Python is executed here except through ``python_test_result``.
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
import json
import random
import re

from postraining.vapo.code_reward import (
    PYTHON_RESULT_CODES,
    PYTHON_REWARD_SCHEMA,
    python_candidate_allowed,
    python_test_result,
)


DATASET_ID = "KodCode/KodCode-Light-RL-10K"
DATASET_REVISION = "dcf78a8bbba9a613b596ce993c4921a38687dfcc"
ADAPTER_SCHEMA = "kodcode_assertion_functions/v1"
_SANDBOX_INSTRUCTIONS = (
    "Your final answer must be a complete executable Python module defining the "
    "requested function, preferably in one ```python code fence. Use deterministic "
    "in-process Python only: no filesystem, process, network, reflection, dynamic "
    "execution, or interactive I/O. Use only ordinary task data fields and "
    "collection/string/math methods; interpreter and frame attributes are "
    "unavailable. Imports are limited to bisect, cmath, collections, datetime, "
    "heapq, itertools, math, re, sys.maxsize, and operator.eq. The verifier has "
    "a restricted attribute allowlist, a 2-second CPU limit, a 3-second wall "
    "limit, and a 384 MiB address-space limit. Hidden assertion tests will call "
    "the function; do not include input-reading or test-running code."
)


class _Ineligible(ValueError):
    """A deterministic, counted dataset exclusion (not infrastructure failure)."""


def _reject(reason: str) -> None:
    raise _Ineligible(reason)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _parse(source: str, reason: str) -> ast.Module:
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        _reject(reason)


def _docstring(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _argument_names(args: ast.arguments) -> list[str]:
    names = [arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    if args.vararg:
        names.append(args.vararg.arg)
    if args.kwarg:
        names.append(args.kwarg.arg)
    return names


def _declaration(source: dict) -> tuple[str, ast.FunctionDef]:
    info = source.get("test_info")
    if not isinstance(info, list) or len(info) != 1 or not isinstance(info[0], dict):
        _reject("invalid_test_info")
    declaration = info[0].get("function_declaration")
    name = info[0].get("function_name")
    if not isinstance(declaration, str) or not isinstance(name, str):
        _reject("invalid_function_declaration")
    tree = _parse(declaration.strip() + "\n    pass\n", "invalid_function_declaration")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        _reject("invalid_function_declaration")
    function = tree.body[0]
    if (
        function.name != name
        or function.decorator_list
        or len(function.body) != 1
        or not isinstance(function.body[0], ast.Pass)
        or name.startswith("_")
    ):
        _reject("invalid_function_declaration")
    names = _argument_names(function.args)
    if len(names) != len(set(names)):
        _reject("invalid_function_signature")
    return declaration.strip(), function


def _adapt_tests(test: str, target: str) -> dict:
    tree = _parse(test, "test_syntax_error")
    functions: list[ast.FunctionDef] = []
    adapted: list[ast.stmt] = []
    bindings: set[str] = {target}
    solution_imports = 0
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "solution"
            and not node.level
        ):
            for alias in node.names:
                if alias.name != target:
                    _reject("unsupported_solution_import")
                solution_imports += 1
                bound = alias.asname or alias.name
                if bound != target:
                    if bound in bindings or bound.startswith("_"):
                        _reject("test_binding_collision")
                    bindings.add(bound)
                    adapted.append(
                        ast.Assign(
                            targets=[ast.Name(id=bound, ctx=ast.Store())],
                            value=ast.Name(id=target, ctx=ast.Load()),
                        )
                    )
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if not python_candidate_allowed(ast.unparse(node)):
                _reject("unsupported_test_import")
            for alias in node.names:
                bound = alias.asname or (
                    alias.name.split(".", 1)[0]
                    if isinstance(node, ast.Import)
                    else alias.name
                )
                if bound in bindings:
                    _reject("test_binding_collision")
                bindings.add(bound)
            adapted.append(node)
            continue
        if _docstring(node):
            adapted.append(node)
            continue
        if not isinstance(node, ast.FunctionDef):
            _reject("unsupported_test_module_statement")
        if not node.name.startswith("test_"):
            _reject("unsupported_test_helper")
        if node.name in bindings:
            _reject("duplicate_test_or_binding")
        bindings.add(node.name)
        if node.decorator_list:
            _reject("unsupported_test_decorator")
        if _argument_names(node.args) or node.args.defaults or node.args.kw_defaults:
            _reject("unsupported_test_fixture_or_signature")
        if node.returns is not None or getattr(node, "type_params", []):
            _reject("unsupported_test_signature")
        # Each invoked test must contain an unconditional assertion. Conditional
        # assertions alone can silently exercise no cases, even with a passing reference.
        if not any(isinstance(statement, ast.Assert) for statement in node.body):
            _reject("nonasserting_test_function")
        for child in ast.walk(node):
            if child is node:
                continue
            if (
                isinstance(child, ast.stmt)
                and not isinstance(
                    child,
                    (
                        ast.Assign,
                        ast.AnnAssign,
                        ast.AugAssign,
                        ast.Assert,
                        ast.If,
                        ast.For,
                    ),
                )
                and not _docstring(child)
            ):
                _reject("unsupported_test_body_statement")
            if isinstance(
                child, (ast.Await, ast.Yield, ast.YieldFrom, ast.Lambda, ast.NamedExpr)
            ):
                _reject("unsupported_test_expression")
            if isinstance(child, ast.Name) and isinstance(
                child.ctx, (ast.Store, ast.Del)
            ):
                if child.id in bindings or child.id == target:
                    _reject("test_rebinds_callable_or_import")
        functions.append(node)
        adapted.append(node)
    if not functions:
        _reject("empty_test_suite")
    # Imports must explicitly identify the entry point; no guessing a module shim.
    if solution_imports != 1:
        _reject("missing_or_duplicate_solution_import")
    # Reject forward-name shadowing too (the binding may occur after a function).
    for function in functions:
        for child in ast.walk(function):
            if isinstance(child, ast.Name) and isinstance(
                child.ctx, (ast.Store, ast.Del)
            ):
                if child.id in bindings:
                    _reject("test_rebinds_callable_or_import")
    adapted_tree = ast.fix_missing_locations(ast.Module(body=adapted, type_ignores=[]))
    setup = ast.unparse(adapted_tree)
    if not python_candidate_allowed(setup):
        _reject("test_policy_rejected")
    return {
        "schema": PYTHON_REWARD_SCHEMA,
        # The only solution import the adapter admits.
        "entry_points": [target],
        "test_setup": [setup],
        "tests": [f"{function.name}()" for function in functions],
    }


def _adapt_row(source: dict) -> tuple[dict, str]:
    for field in ("question_id", "question", "solution", "test", "subset"):
        if not isinstance(source.get(field), str) or not source[field].strip():
            _reject(f"invalid_{field}")
    if source.get("style") != "instruct":
        _reject("unsupported_question_style")
    declaration, target = _declaration(source)
    reference = source["solution"].strip()
    reference_tree = _parse(reference, "reference_syntax_error")
    definitions = [
        node for node in reference_tree.body if isinstance(node, ast.FunctionDef)
    ]
    if len(definitions) != 1 or definitions[0].name != target.name:
        _reject("unsupported_reference_entrypoint")
    definition = definitions[0]
    if definition.decorator_list or any(
        isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(reference_tree)
    ):
        _reject("unsupported_reference_definition")
    if ast.dump(definition.args) != ast.dump(target.args):
        _reject("reference_signature_mismatch")
    verification = _adapt_tests(source["test"], target.name)
    question = source["question"].strip()
    row = {
        "question_id": source["question_id"],
        "subset": source["subset"],
        "difficulty": str(source.get("gpt_difficulty") or "unknown"),
        "question_sha256": _sha256(" ".join(question.split())),
        "prompt": [
            {
                "role": "user",
                "content": (
                    question
                    + "\n\nRequired function declaration:\n"
                    + declaration
                    + "\n\n"
                    + _SANDBOX_INSTRUCTIONS
                ),
            }
        ],
        "verification_info": verification,
    }
    return row, reference


def prepare_rows(
    dataset_revision: str, *, problems: int, seed: int
) -> tuple[list[dict], dict]:
    """Load an immutable revision, scan syntax, then preflight a shuffled prefix.

    Selection never consults model outcomes or dataset success labels. Reference
    failure excludes a task from this sandbox's supported population, not from a
    benchmark score. The unscanned remainder is not claimed reference-eligible.
    """
    if not isinstance(dataset_revision, str) or not re.fullmatch(
        r"[0-9a-f]{40}", dataset_revision
    ):
        raise ValueError(
            "dataset_revision must be a full lowercase 40-character commit SHA"
        )
    if isinstance(problems, bool) or not isinstance(problems, int) or problems < 1:
        raise ValueError("problems must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_ID, "default", split="train", revision=dataset_revision
    )
    reasons: Counter = Counter()
    population_subsets: Counter = Counter()
    population_difficulties: Counter = Counter()
    eligible: list[tuple[dict, str]] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    digest = hashlib.sha256()
    for source in dataset:
        digest.update((_canonical(source) + "\n").encode("utf-8"))
        population_subsets[str(source.get("subset") or "unknown")] += 1
        population_difficulties[str(source.get("gpt_difficulty") or "unknown")] += 1
        question_id = source.get("question_id")
        question = source.get("question")
        question_hash = (
            _sha256(" ".join(question.split())) if isinstance(question, str) else None
        )
        if isinstance(question_id, str) and question_id in seen_ids:
            reasons["duplicate_question_id"] += 1
            continue
        if question_hash is not None and question_hash in seen_questions:
            reasons["duplicate_question_text"] += 1
            continue
        if isinstance(question_id, str):
            seen_ids.add(question_id)
        if question_hash is not None:
            seen_questions.add(question_hash)
        try:
            eligible.append(_adapt_row(source))
        except _Ineligible as error:
            reasons[str(error)] += 1
    # Sorting makes the seeded order independent of Arrow shard traversal order.
    eligible.sort(key=lambda item: item[0]["question_id"])
    random.Random(seed).shuffle(eligible)
    selected: list[dict] = []
    preflights: list[dict] = []
    reference_rejections: Counter = Counter()
    for row, reference in eligible:
        result = python_test_result(reference, row["verification_info"])
        preflights.append({"question_id": row["question_id"], "result": result})
        if result == "pass":
            selected.append(row)
            if len(selected) == problems:
                break
        else:
            reference_rejections[result] += 1
    manifest = {
        "adapter_schema": ADAPTER_SCHEMA,
        "verifier_schema": PYTHON_REWARD_SCHEMA,
        "dataset_id": DATASET_ID,
        "dataset_revision": dataset_revision,
        "dataset_config": "default",
        "dataset_split": "train",
        "dataset_sha256": digest.hexdigest(),
        "dataset_hash_encoding": "SHA256 of each complete source row as sorted compact UTF-8 JSON plus LF, in dataset order",
        "license": "CC-BY-NC-4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
        "dataset_url": f"https://huggingface.co/datasets/{DATASET_ID}/tree/{dataset_revision}",
        "seed": seed,
        "requested_problems": problems,
        "population_count": len(dataset),
        "structurally_eligible_count": len(eligible),
        "structurally_rejected_count": sum(reasons.values()),
        "structural_rejection_counts": dict(sorted(reasons.items())),
        "reference_scanned_count": len(preflights),
        "reference_unscanned_count": len(eligible) - len(preflights),
        "reference_pass_count": len(selected),
        "reference_rejection_counts": dict(sorted(reference_rejections.items())),
        "reference_preflights": preflights,
        "selected_count": len(selected),
        "sample_complete": len(selected) == problems,
        "sampled_ids": [row["question_id"] for row in selected],
        "selected_rows_sha256": _sha256(_canonical(selected)),
        "population_subsets": dict(sorted(population_subsets.items())),
        "population_difficulties": dict(sorted(population_difficulties.items())),
        "selected_subsets": dict(
            sorted(Counter(row["subset"] for row in selected).items())
        ),
        "selected_difficulties": dict(
            sorted(Counter(row["difficulty"] for row in selected).items())
        ),
        "selection_rationale": (
            "Scan all rows structurally, retaining the first occurrence of each ID and "
            "whitespace-normalized question (including rejected first occurrences); "
            "sort eligible rows by question_id, shuffle with Python Random(seed), and "
            "sandbox-preflight references in that order until requested_problems pass "
            "or the finite population is exhausted. No generated-model success filtering, "
            "difficulty balancing, or use of gpt/r1 success labels. Unscanned references "
            "have unknown executability; this is a sandbox-compatible subset, not the full dataset."
        ),
        "supported_test_contract": (
            "One declared synchronous reference function with matching argument AST; "
            "one explicit from solution import target (aliases are bound explicitly). "
            "Module statements are allowed imports, docstrings, and unique test_* "
            "functions with no parameters, annotations, or decorators. Each test has "
            "at least one direct unconditional assert. Bodies allow assignments, asserts, "
            "if/for statements, and docstrings under the existing candidate syntax policy. "
            "All definitions and bodies are preserved and every test is explicitly called "
            "in source order. Helpers, fixtures, pytest imports/features, classes, async, "
            "try/with, return/raise, generators, lambda, named expressions, and rebinding "
            "imported/test/target names are excluded, not erased."
        ),
        "limitations": [
            "Reference preflight checks executable assertions, not test completeness or specification correctness.",
            "Structural exclusions are exclusive first-failure counts, not counts of every defect per row.",
            "Difficulty is the supplied gpt_difficulty label and is not used for selection.",
            "Known dataset upstream filtering and possible model pretraining overlap prevent claims of uncontaminated generalization.",
        ],
    }
    return selected, manifest


def extract_code(text: str) -> str | None:
    """Use only completed final-answer text; never recover an unfinished scratchpad.

    If a chat template supplies the opening think tag outside generated tokens,
    the caller must retain that tag when scoring an unfinished thinking response.
    A lone closing tag is supported for completed template-prefilled thinking.
    """
    if not isinstance(text, str):
        return None
    depth = 0
    saw_marker = False
    final_start = 0
    for marker in re.finditer(r"</?think>", text):
        if marker.group() == "<think>":
            depth += 1
        elif depth:
            depth -= 1
            if not depth:
                final_start = marker.end()
        elif not saw_marker:
            # The template, rather than generated text, supplied this opening.
            final_start = marker.end()
        else:
            return None
        saw_marker = True
    if depth:
        return None
    final = text[final_start:].strip()
    if not final:
        return None
    # Parse fence boundaries rather than searching through an unsupported fence's
    # contents. An unfinished final fence must not fall back to an earlier snippet.
    fence = re.compile(r"^[ \t]*(`{3,}|~{3,})([^\r\n]*)[ \t]*$", re.MULTILINE)
    active: tuple[str, int, int, bool] | None = None
    candidates: list[str] = []
    saw_fence = False
    for match in fence.finditer(final):
        marker, label = match.group(1), match.group(2).strip().lower()
        if active is None:
            saw_fence = True
            active = (
                marker[0],
                len(marker),
                match.end(),
                label in ("", "python", "py", "python3"),
            )
        elif marker[0] == active[0] and len(marker) >= active[1] and not label:
            if active[3]:
                candidates.append(final[active[2] : match.start()].strip())
            active = None
    if active is not None:
        return None
    if candidates:
        return candidates[-1] or None
    if saw_fence:
        return None
    try:
        module = ast.parse(final)
    except (SyntaxError, ValueError, RecursionError):
        return None
    return final if module.body else None


def score_completion(text: str, verification_info: dict) -> str:
    """Return the existing verifier labels without masking sandbox failures."""
    if verification_info.get("schema") != PYTHON_REWARD_SCHEMA:
        raise ValueError("incompatible Python verifier schema")
    code = extract_code(text)
    return (
        "format_ineligible"
        if code is None
        else python_test_result(code, verification_info)
    )


def _summary(attempts: list[dict], samples_per_problem: int) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    results: Counter = Counter()
    for attempt in attempts:
        groups[attempt["question_id"]].append(attempt)
        results[attempt["result"]] += 1
    total = len(attempts)
    problems = len(groups)
    successes = sum(attempt["correct"] for attempt in attempts)
    expected = problems * samples_per_problem
    complete_histogram: Counter = Counter()
    incomplete_histogram: Counter = Counter()
    per_problem = []
    for question_id, group in sorted(groups.items()):
        count = sum(attempt["correct"] for attempt in group)
        complete = len(group) == samples_per_problem
        (complete_histogram if complete else incomplete_histogram)[count] += 1
        per_problem.append(
            {
                "question_id": question_id,
                "attempts": len(group),
                "successes": count,
                "complete": complete,
                "missing_sample_indices": sorted(
                    set(range(samples_per_problem)) - {a["sample_index"] for a in group}
                ),
            }
        )
    complete_count = sum(complete_histogram.values())
    any_success = sum(row["successes"] > 0 for row in per_problem)
    all_fail = complete_histogram[0]
    all_pass = complete_histogram[samples_per_problem]
    mixed = complete_count - all_fail - all_pass
    truncated = sum(not attempt["terminated"] for attempt in attempts)
    ratio = lambda numerator, denominator: (
        numerator / denominator if denominator else None
    )
    return {
        "attempt_count": total,
        "problem_count": problems,
        "samples_per_problem": samples_per_problem,
        "expected_attempt_count_for_observed_problems": expected,
        "missing_attempt_count_for_observed_problems": expected - total,
        "complete_problem_count": complete_count,
        "incomplete_problem_count": problems - complete_count,
        "success_count": successes,
        "observed_per_attempt_accuracy": ratio(successes, total),
        "success_fraction_of_expected_attempts": ratio(successes, expected),
        "observed_any_success_count": any_success,
        "observed_any_success_fraction": ratio(any_success, problems),
        "all_fail_problem_count": all_fail,
        "all_pass_problem_count": all_pass,
        "mixed_outcome_problem_count": mixed,
        "all_fail_fraction": ratio(all_fail, problems),
        "all_pass_fraction": ratio(all_pass, problems),
        "mixed_outcome_fraction": ratio(mixed, problems),
        "incomplete_fraction": ratio(problems - complete_count, problems),
        "per_problem_success_histogram": dict(
            sorted(
                (str(k), complete_histogram[k] + incomplete_histogram[k])
                for k in complete_histogram.keys() | incomplete_histogram.keys()
            )
        ),
        "complete_problem_success_histogram": dict(
            sorted((str(k), v) for k, v in complete_histogram.items())
        ),
        "incomplete_problem_success_histogram": dict(
            sorted((str(k), v) for k, v in incomplete_histogram.items())
        ),
        "per_problem": per_problem,
        "result_counts": dict(sorted(results.items())),
        "truncation_count": truncated,
        "truncation_rate": ratio(truncated, total),
        "policy_rejection_count": results["policy_rejected"],
        "policy_rejection_rate": ratio(results["policy_rejected"], total),
        "format_ineligible_count": results["format_ineligible"],
        "format_ineligible_rate": ratio(results["format_ineligible"], total),
        "mean_generated_tokens": ratio(sum(a["tokens"] for a in attempts), total),
    }


def summarize(attempts: list[dict], *, samples_per_problem: int) -> dict:
    """Describe observations, retaining failures and incomplete groups in denominators.

    Entirely absent problems cannot be inferred from this interface: the evaluator
    must also report expected IDs and zero-attempt IDs from its selected manifest.
    """
    if (
        isinstance(samples_per_problem, bool)
        or not isinstance(samples_per_problem, int)
        or samples_per_problem < 1
    ):
        raise ValueError("samples_per_problem must be a positive integer")
    seen: set[tuple[str, int]] = set()
    metadata: dict[str, tuple[str, str]] = {}
    for attempt in attempts:
        question_id = attempt["question_id"]
        index = attempt["sample_index"]
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("attempt question_id must be a nonempty string")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < samples_per_problem
        ):
            raise ValueError("sample_index must be in [0, samples_per_problem)")
        if (question_id, index) in seen:
            raise ValueError("duplicate question_id/sample_index")
        seen.add((question_id, index))
        if not isinstance(attempt["correct"], bool) or not isinstance(
            attempt["terminated"], bool
        ):
            raise ValueError("correct and terminated must be booleans")
        if attempt["result"] not in PYTHON_RESULT_CODES:
            raise ValueError("unknown verifier result label")
        if attempt["correct"] != (attempt["result"] == "pass"):
            raise ValueError("correct must agree with the verifier result")
        if not isinstance(attempt["text"], str):
            raise ValueError("attempt text must be a string")
        if (
            isinstance(attempt["tokens"], bool)
            or not isinstance(attempt["tokens"], int)
            or attempt["tokens"] < 0
        ):
            raise ValueError("attempt tokens must be a nonnegative integer")
        labels = (attempt["subset"], attempt["difficulty"])
        if not all(isinstance(label, str) for label in labels):
            raise ValueError("subset and difficulty must be strings")
        if metadata.setdefault(question_id, labels) != labels:
            raise ValueError("inconsistent problem metadata across samples")
    result = _summary(attempts, samples_per_problem)
    for field in ("difficulty", "subset"):
        by_label: dict[str, list[dict]] = defaultdict(list)
        for attempt in attempts:
            by_label[attempt[field]].append(attempt)
        result[f"by_{field}"] = {
            label: _summary(rows, samples_per_problem)
            for label, rows in sorted(by_label.items())
        }
    result["metric_contract"] = {
        "observed_per_attempt_accuracy": "passes / all supplied attempts, including truncation, format, and policy failures",
        "observed_any_success_fraction": "observed problems with >=1 pass / all observed problems; not Avg@k or a pass@k estimator",
        "outcome_fractions": "complete all-fail/all-pass/mixed group counts / all observed problems; incomplete groups are a separate category",
        "missing_attempts": "expected slots minus observations, for observed IDs only; missing slots are not classified as failures",
        "absent_problems": "IDs with zero attempts are unknowable from attempts alone; join with the selected manifest, never claim full-population completion from these metrics",
        "truncation": "terminated is false; overlaps result categories and is not excluded",
    }
    return result
