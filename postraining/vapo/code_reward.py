"""Binary, sandboxed execution rewards for Python programming tasks."""

from __future__ import annotations

import ast
import json
import os
import secrets
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache


PYTHON_REWARD_SCHEMA = "bwrap_python_safe_ast_all_tests_binary/v2"
PYTHON_RESULT_CODES = {
    "format_ineligible": 0,
    "pass": 1,
    "policy_rejected": 2,
    "tests_failed": 3,
    "timeout": 4,
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
    "quit",
    "setattr",
    "delattr",
    "vars",
}
FORBIDDEN_ATTRIBUTES = {
    "__class__",
    "__code__",
    "__dict__",
    "__getattribute__",
    "__globals__",
    "__mro__",
    "__subclasses__",
    "exit",
    "modules",
    "setprofile",
    "settrace",
    "stderr",
    "stdin",
    "stdout",
}
SAFE_DUNDER_ATTRIBUTES = {"__add__", "__contains__"}


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
    for node in ast.walk(tree):
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
        elif isinstance(node, ast.Attribute):
            if (
                (
                    node.attr.startswith("_")
                    and node.attr not in SAFE_DUNDER_ATTRIBUTES
                )
                or node.attr in FORBIDDEN_ATTRIBUTES
                or (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "sys"
                    and node.attr != "maxsize"
                )
            ):
                return False
    return True


@lru_cache(maxsize=1)
def _nproc_limit() -> int:
    """Allow a small sandbox budget above this busy user's current threads."""
    uid = os.getuid()
    tasks = 0
    with os.scandir("/proc") as processes:
        for process in processes:
            if not process.name.isdigit():
                continue
            try:
                with open(
                    f"/proc/{process.name}/status", encoding="utf-8"
                ) as status_file:
                    status = status_file.read()
                real_uid = int(
                    next(
                        line for line in status.splitlines()
                        if line.startswith("Uid:")
                    ).split()[1]
                )
                if real_uid == uid:
                    tasks += len(os.listdir(f"/proc/{process.name}/task"))
            except (FileNotFoundError, PermissionError, StopIteration, ValueError):
                continue
    return tasks + 64


def python_test_result(code: str, verification_info: dict) -> str:
    """Run one completion with no network or writable host filesystem."""
    bwrap = shutil.which("bwrap")
    prlimit = shutil.which("prlimit")
    python = "/usr/bin/python3"
    if bwrap is None or prlimit is None or not os.path.exists(python):
        raise RuntimeError("Python rewards require bwrap, prlimit, and python3")
    tests = verification_info.get("tests")
    imports = verification_info.get("test_setup") or []
    if not isinstance(tests, list) or not tests:
        raise ValueError("Python verifier requires a nonempty test list")
    if not isinstance(imports, list):
        raise ValueError("Python verifier test_setup must be a list")
    sentinel = secrets.token_bytes(32)
    candidate = normalize_python_answer(code)
    if not python_candidate_allowed(candidate):
        return "policy_rejected"
    # Candidate code and tests execute in their own namespace. The harness
    # keeps private references to exec/write and emits a per-process sentinel
    # only after every assertion ran. Thus SystemExit is a failure and an
    # uncatchable os._exit(0) cannot turn an early exit into reward 1.
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
            "def _harness_import(name, globals=None, locals=None, fromlist=(), level=0):",
            "    if name.split('.', 1)[0] == 'sys':",
            "        return _harness_sys_proxy",
            "    return _harness_real_import(name, globals, locals, fromlist, level)",
            "_candidate_builtins = vars(_harness_builtins).copy()",
            "_candidate_builtins['__import__'] = _harness_import",
            "_candidate_globals = {'__builtins__': _candidate_builtins}",
            "try:",
            *[
                f"    _harness_exec({source!r}, _candidate_globals)"
                for source in [candidate, *map(str, imports), *map(str, tests)]
            ],
            "except BaseException:",
            "    _harness_os._exit(20)",
            f"_harness_write(1, {sentinel!r})",
        ]
    )
    command = [
        prlimit,
        "--cpu=2:2",
        f"--as={384 << 20}:{384 << 20}",
        f"--fsize={1 << 20}:{1 << 20}",
        "--nofile=32:32",
        f"--nproc={_nproc_limit()}:{_nproc_limit()}",
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
        "--setenv", "HOME", "/tmp",
        "--setenv", "PYTHONHASHSEED", "0",
        python,
        "-I",
        "-S",
        "-c",
        program,
    ]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3.0,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "timeout"
    except OSError as error:
        raise RuntimeError("Python verifier infrastructure failed") from error
    if completed.returncode == 0 and completed.stdout.endswith(sentinel):
        return "pass"
    if completed.stderr.startswith((b"bwrap:", b"prlimit:")):
        raise RuntimeError(
            "Python verifier sandbox failed: "
            + completed.stderr.decode(errors="replace")[:500]
        )
    return "tests_failed"


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
