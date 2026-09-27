"""Sandboxed binary and test-fraction rewards for Python programming tasks."""

from __future__ import annotations

import ast
import json
import os
import secrets
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


# v6: tests run in their own namespace over pristine builtins and receive only
# the row's declared ``entry_points`` from the candidate; candidate imports
# bind private module copies whose classes must stay unmodified; attribute or
# item stores rooted at an imported name are rejected; output is discarded so
# printing cannot corrupt the sentinel; PYTHONHASHSEED actually applies.
# Under v5 candidate and tests shared one namespace, so a module-level
# ``abs = lambda x: 0`` or ``math.isclose = ...`` could satisfy tolerance
# asserts without solving the task.
PYTHON_REWARD_SCHEMA = "bwrap_python_positive_ast_isolated_tests_binary/v6"
# The immutable pool/test contract above is unchanged. This separately binds
# the learning objective when a run rewards the fraction of tests passed.
PYTHON_PARTIAL_REWARD_SCHEMA = "bwrap_python_isolated_test_fraction/v2"
PYTHON_RESULT_CODES = {
    "not_applicable": 0,
    "format_ineligible": 1,
    "pass": 2,
    "policy_rejected": 3,
    "tests_failed": 4,
    "timeout": 5,
}

SAFE_IMPORT_ROOTS = {
    "bisect",
    "cmath",
    "collections",
    "datetime",
    "heapq",
    "itertools",
    "math",
    "operator",
    "re",
    "sys",
}
FORBIDDEN_NAMES = {
    "SystemExit",
    "__builtins__",
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "locals",
    "open",
    "property",
    "quit",
    "setattr",
    "staticmethod",
    "classmethod",
    "delattr",
    "type",
    "vars",
}
SAFE_ATTRIBUTES = {
    "ChainMap", "__add__", "__contains__", "a", "add", "append", "b",
    "bisect_left", "bisect_right", "ceil", "clear", "compile", "count",
    "data", "date", "e", "elements", "end", "extend", "findall",
    "finditer", "floor", "from_iterable", "get", "groupby", "heapify",
    "heappop", "heappush", "intersection", "isalpha", "isdigit",
    "islower", "issubset", "isupper", "items", "join", "keys", "left",
    "log", "log10", "log2", "lower", "match", "maxsize", "merge",
    "most_common", "nlargest", "nsmallest", "pattern", "pi", "pop", "pow",
    "re", "remove", "replace", "right", "search", "setdefault", "sort",
    "split", "sqrt", "start", "strip", "sub", "upper", "values",
}
_SANDBOX_SLOTS = threading.BoundedSemaphore(8)


@dataclass(frozen=True)
class PythonTestScore:
    passed: int
    total: int
    status: str

    @property
    def reward(self) -> float:
        return self.passed / self.total


def python_test_cases(tests: list[str]) -> tuple[list[list[str]], list[str]]:
    """Group ordered fixture statements with their next assertion block.

    A compound statement containing assertions or an expression call to a
    known assertion helper is one case, irrespective of reached branches.
    Definitions never earn credit; helper dependencies are resolved to a
    fixed point. Assignment calls remain fixtures for their next assertion.
    Trailing fixtures remain executable but never earn credit.
    """
    trees = []
    for source in tests:
        if not isinstance(source, str):
            raise ValueError("fractional Python tests must be strings")
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError) as error:
            raise ValueError("fractional Python tests must be valid Python") from error
        trees.append(tree)

    def executing_nodes(node):
        # Defining a helper does not execute its body. Class bodies do run,
        # but their nested methods/lambdas still must not manufacture credit.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return
        yield node
        for child in ast.iter_child_nodes(node):
            yield from executing_nodes(child)

    helpers = {
        node.name: node
        for tree in trees for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and not any(isinstance(child, (ast.Yield, ast.YieldFrom))
                    for statement in node.body for child in executing_nodes(statement))
    }
    helper_checks = {
        name for name, function in helpers.items()
        if any(isinstance(node, ast.Assert)
               for statement in function.body for node in executing_nodes(statement))
    }
    while True:
        expanded = helper_checks | {
            name for name, function in helpers.items()
            if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in helper_checks
                   for statement in function.body for node in executing_nodes(statement))
        }
        if expanded == helper_checks:
            break
        helper_checks = expanded

    cases = []
    pending = []
    for source, tree in zip(tests, trees):
        pending.append(source)
        assertion = any(isinstance(node, ast.Assert) for node in executing_nodes(tree))
        helper_call = any(
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id in helper_checks
            for node in executing_nodes(tree)
        )
        if assertion or helper_call:
            cases.append(pending)
            pending = []
    if not cases:
        raise ValueError("fractional Python tests must contain an assertion")
    return cases, pending


def normalize_python_answer(answer: str) -> str:
    text = answer.strip()
    if text.startswith("```python"):
        text = text[len("```python"):]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def python_candidate_allowed(code: str) -> bool:
    """Constrain MBPP modules to ordinary, deterministic solution code.

    Bubblewrap protects the host, while this syntax policy protects reward
    integrity: candidate code cannot terminate or introspect the verifier
    process and forge its success control path. The import allowlist covers
    every module used by the official MBPP training solutions.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError):
        return False
    imported = {
        alias.asname or alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            # Patching an imported module or class (``math.sqrt = ...``,
            # ``Counter.most_common = ...``) mutates objects the tests may
            # share; candidate code never needs to.
            root = node.value
            while isinstance(root, (ast.Attribute, ast.Subscript)):
                root = root.value
            if isinstance(root, ast.Name) and root.id in imported:
                return False
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in SAFE_IMPORT_ROOTS:
                    return False
                if root == "sys" and alias.asname not in (None, "sys"):
                    return False
                if root == "operator":
                    return False
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if node.level or root not in SAFE_IMPORT_ROOTS:
                return False
            imported = {alias.name for alias in node.names}
            if "*" in imported or any(name.startswith("_") for name in imported):
                return False
            if root == "sys" and imported - {"maxsize"}:
                return False
            if root == "operator" and imported - {"eq"}:
                return False
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                return False
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("__") and node.name != "__init__":
                return False
        elif isinstance(node, ast.Attribute):
            if node.attr not in SAFE_ATTRIBUTES or (
                isinstance(node.ctx, ast.Store) and node.attr.startswith("__")
            ):
                return False
    return True


def _python_test_score(
    code: str, verification_info: dict, *, require_assertions: bool,
) -> PythonTestScore:
    """Run one completion with no network or writable host filesystem.

    ``verification_info["entry_points"]`` names the candidate definitions the
    tests call; nothing else the candidate binds is visible to them.
    """
    bwrap = shutil.which("bwrap")
    prlimit = shutil.which("prlimit")
    python = "/usr/bin/python3"
    if bwrap is None or prlimit is None or not os.path.exists(python):
        raise RuntimeError("Python rewards require bwrap, prlimit, and python3")
    tests = verification_info.get("tests")
    imports = verification_info.get("test_setup") or []
    if not isinstance(tests, list) or not tests:
        raise ValueError("Python verifier requires a nonempty test list")
    # Pool tests contain ordered top-level fixture statements as well as
    # assertions. Group fixtures with their following case, so fixture
    # failures fail that case without skipping later independent cases.
    if require_assertions:
        tests, trailing_fixtures = python_test_cases(tests)
    else:
        tests = [[str(test)] for test in tests]
        trailing_fixtures = []
    total = len(tests)
    if not isinstance(imports, list):
        raise ValueError("Python verifier test_setup must be a list")
    entry_points = verification_info.get("entry_points")
    if (
        not isinstance(entry_points, list)
        or not entry_points
        or not all(
            isinstance(name, str) and name.isidentifier() and not name.startswith("__")
            for name in entry_points
        )
    ):
        raise ValueError("Python verifier requires nonempty non-dunder entry_points")
    sentinel = secrets.token_bytes(32)
    candidate = normalize_python_answer(code)
    if not python_candidate_allowed(candidate):
        return PythonTestScore(0, total, "policy_rejected")
    # The candidate executes over a copy of builtins whose importer returns
    # private module copies; the tests execute afterwards in a separate
    # namespace over pristine builtins holding only the declared entry
    # points. Classes of every module the candidate imports are snapshotted
    # and must be unchanged after the candidate and after each test
    # statement, so aliasing an imported class cannot patch what the tests
    # call. The harness keeps private references to exec/write and emits a
    # per-process sentinel only after every test entry ran. Thus SystemExit is
    # a failure and an uncatchable os._exit(0) cannot turn an early exit into
    # reward 1.
    program = "\n".join(
        [
            "import os as _harness_os",
            "import builtins as _harness_builtins",
            "import sys as _harness_sys",
            "import types as _harness_types",
            "_harness_exec = exec",
            "_harness_write = _harness_os.write",
            "_harness_real_import = _harness_builtins.__import__",
            "_harness_sys_proxy = _harness_types.SimpleNamespace(maxsize=_harness_sys.maxsize)",
            # Candidate and test output must not reach the sentinel stream.
            "_harness_sys.stdout = open(_harness_os.devnull, 'w')",
            "_harness_copies = {}",
            "_harness_classes = {}",
            "def _harness_snapshot(module):",
            "    for value in list(vars(module).values()):",
            "        if isinstance(value, type) and id(value) not in _harness_classes:",
            "            _harness_classes[id(value)] = (value, dict(vars(value)))",
            "def _harness_check():",
            "    for cls, before in _harness_classes.values():",
            "        after = vars(cls)",
            "        if after.keys() != before.keys() or any(after[k] is not v for k, v in before.items()):",
            "            raise AssertionError('library class mutated')",
            "def _harness_import(name, globals=None, locals=None, fromlist=(), level=0):",
            "    if name.split('.', 1)[0] == 'sys':",
            "        return _harness_sys_proxy",
            "    module = _harness_real_import(name, globals, locals, fromlist, level)",
            "    _harness_snapshot(module)",
            "    for item in fromlist or ():",
            "        value = getattr(module, item, None)",
            "        if isinstance(value, _harness_types.ModuleType):",
            "            _harness_snapshot(value)",
            "    private = _harness_copies.get(id(module))",
            "    if private is None:",
            "        private = _harness_copies[id(module)] = _harness_types.ModuleType(module.__name__)",
            "    for key, value in vars(module).items():",
            "        private.__dict__.setdefault(key, value)",
            "    return private",
            "_test_globals = {'__builtins__': vars(_harness_builtins).copy()}",
            "_candidate_builtins = vars(_harness_builtins).copy()",
            "_candidate_builtins['__import__'] = _harness_import",
            "_candidate_globals = {'__builtins__': _candidate_builtins}",
            "_harness_passed = 0",
            "try:",
            f"    _harness_exec({candidate!r}, _candidate_globals)",
            "    _harness_check()",
            f"    for _harness_name in {list(entry_points)!r}:",
            "        _test_globals[_harness_name] = _candidate_globals[_harness_name]",
            *[
                line
                for source in map(str, imports)
                for line in (
                    f"    _harness_exec({source!r}, _test_globals)",
                    "    _harness_check()",
                )
            ],
            f"    for _harness_case in {tests!r}:",
            "        _harness_test_ok = True",
            "        for _harness_source in _harness_case:",
            "            try:",
            "                _harness_exec(_harness_source, _test_globals)",
            "            except BaseException:",
            "                _harness_test_ok = False" if require_assertions else "                raise",
            # Integrity checks stay OUTSIDE the per-test catch: mutation
            # invalidates the whole completion, including earlier passes.
            "            _harness_check()",
            "            if not _harness_test_ok:",
            "                break",
            "        if _harness_test_ok:",
            "            _harness_passed += 1",
            *[
                line
                for source in trailing_fixtures
                for line in (
                    f"    _harness_exec({source!r}, _test_globals)",
                    "    _harness_check()",
                )
            ],
            "except BaseException:",
            "    _harness_os._exit(20)",
            f"_harness_write(1, {sentinel!r} + b':' + str(_harness_passed).encode('ascii'))",
        ]
    )
    command = [
        prlimit,
        "--cpu=2:2",
        f"--as={384 << 20}:{384 << 20}",
        f"--fsize={1 << 20}:{1 << 20}",
        "--nofile=32:32",
        "--",
        bwrap,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind-try", "/lib64", "/lib64",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        "--chdir", "/tmp",
        # ``-I`` would imply ``-E`` and ignore PYTHONHASHSEED, leaving set
        # iteration order (and so some verdicts) random per run. Clear the
        # environment in bwrap instead and keep -I's other isolation flags.
        "--clearenv",
        "--setenv", "HOME", "/tmp",
        "--setenv", "PYTHONHASHSEED", "0",
        python,
        "-s",
        "-P",
        "-S",
        "-c",
        program,
    ]
    try:
        with _SANDBOX_SLOTS:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=3.0,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return PythonTestScore(0, total, "timeout")
    except OSError as error:
        raise RuntimeError("Python verifier infrastructure failed") from error
    prefix = sentinel + b":"
    if completed.returncode == 0 and completed.stdout.startswith(prefix):
        payload = completed.stdout[len(prefix):]
        if payload.isdigit():
            passed = int(payload)
            if 0 <= passed <= total:
                return PythonTestScore(passed, total, "pass" if passed == total else "tests_failed")
    if completed.stderr.startswith((b"bwrap:", b"prlimit:")):
        raise RuntimeError(
            "Python verifier sandbox failed: "
            + completed.stderr.decode(errors="replace")[:500]
        )
    return PythonTestScore(0, total, "tests_failed")


def python_test_score(code: str, verification_info: dict) -> PythonTestScore:
    """Grade every immutable test case; failures do not skip later cases.

    The denominator is the count of assertion-bearing test blocks, not
    the number reached, executed assertions, or candidate-generated values.
    Timeout, candidate/setup failure and sandbox-integrity failure zero the
    whole result. Fixtures preceding a test belong to that case. Pool defects
    raise instead of silently changing the denominator.
    """
    return _python_test_score(code, verification_info, require_assertions=True)


def python_test_result(code: str, verification_info: dict) -> str:
    """Backward-compatible all-tests-pass status, including legacy fixtures."""
    return _python_test_score(code, verification_info, require_assertions=False).status


def python_tests_pass(code: str, verification_info: dict) -> bool:
    return python_test_result(code, verification_info) == "pass"


def batch_python_tests_pass(
    answers: list[str], verification_info: dict, workers: int = 8
) -> list[bool]:
    # Copy through JSON once so no mutable Arrow-backed object is shared with
    # worker subprocess preparation.
    info = json.loads(json.dumps(verification_info))
    with ThreadPoolExecutor(max_workers=min(workers, len(answers))) as pool:
        return list(pool.map(lambda answer: python_tests_pass(answer, info), answers))


def batch_python_test_results(
    answers: list[str], verification_info: dict, workers: int = 8
) -> list[str]:
    info = json.loads(json.dumps(verification_info))
    with ThreadPoolExecutor(max_workers=min(workers, len(answers))) as pool:
        return list(pool.map(lambda answer: python_test_result(answer, info), answers))


def batch_python_test_scores(
    answers: list[str], verification_info: dict, workers: int = 8,
) -> list[PythonTestScore]:
    if workers < 1:
        raise ValueError("Python verifier workers must be positive")
    if not answers:
        return []
    info = json.loads(json.dumps(verification_info))
    with ThreadPoolExecutor(max_workers=min(workers, len(answers))) as pool:
        return list(pool.map(lambda answer: python_test_score(answer, info), answers))
