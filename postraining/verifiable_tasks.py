"""Dataset-independent final-answer and isolated stdin/stdout rewards.

Stdio comparison removes ASCII trailing horizontal whitespace from each line and
trailing empty lines, and normalizes CRLF to LF. Leading/internal whitespace and
all non-whitespace content remain significant. Expected outputs never enter the
sandbox: one input is piped to one fresh interpreter, then compared by the host.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
from functools import lru_cache
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import tempfile
import threading
import time

from postraining import core
from postraining.kodcode_eval import extract_code

VERIFIABLE_TASK_SCHEMA = "verifiable_task/v1"
MAX_TEST_CASES = 128
MAX_TEST_BYTES = 2 << 20
MAX_CASE_BYTES = 256 << 10
MAX_CODE_BYTES = 128 << 10
_CASE_WALL_SECONDS = 3.0
_TOTAL_WALL_SECONDS = 60.0
_OUTPUT_BYTES = 256 << 10
_SANDBOX_SLOTS = threading.BoundedSemaphore(4)
_EOS = re.compile(r"<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>|<\|end_of_text\|>|</s>")


def is_verifiable_task(row: dict) -> bool:
    if not isinstance(row, dict):
        return False
    reward, info = row.get("reward_model"), row.get("verification_info")
    return (
        isinstance(reward, dict)
        and isinstance(reward.get("style"), str)
        and reward["style"].startswith("verifiable_task/")
    ) or (
        isinstance(info, dict)
        and isinstance(info.get("schema"), str)
        and info["schema"].startswith("verifiable_task/")
    )


def validate_verifiable_task(row: dict) -> None:
    """Fail closed on malformed data; never silently drop a hidden test."""
    if not isinstance(row, dict):
        raise ValueError("verifiable task must be a mapping")
    reward = row.get("reward_model")
    if not isinstance(reward, dict) or reward.get("style") != VERIFIABLE_TASK_SCHEMA:
        raise ValueError("invalid verifiable reward schema")
    truth = reward.get("ground_truth")
    if not isinstance(truth, str) or not truth.strip():
        raise ValueError("ground_truth must be a nonempty string")
    prompt = row.get("prompt")
    if (
        not isinstance(prompt, list)
        or len(prompt) != 1
        or not isinstance(prompt[0], dict)
        or prompt[0].get("role") != "user"
        or not isinstance(prompt[0].get("content"), str)
        or not prompt[0]["content"].strip()
    ):
        raise ValueError("invalid user prompt")
    extra = row.get("extra_info")
    if (
        not isinstance(extra, dict)
        or not isinstance(extra.get("domain"), str)
        or not extra["domain"].strip()
    ):
        raise ValueError("domain must be a nonempty reporting label")
    info = row.get("verification_info")
    if (
        not isinstance(info, dict)
        or info.get("schema") != VERIFIABLE_TASK_SCHEMA
        or not isinstance(info.get("kind"), str)
        or info.get("kind") not in {"math", "text", "python_stdio"}
    ):
        raise ValueError("invalid verification_info schema or kind")
    if info["kind"] != "python_stdio":
        if any(
            info.get(key) is not None
            for key in ("inputs", "outputs", "call_type", "fn_name")
        ):
            raise ValueError("noncode task must not contain executable tests")
        return
    if (
        info.get("call_type") != "std"
        or "fn_name" not in info
        or info["fn_name"] is not None
    ):
        raise ValueError("invalid stdio call convention")
    inputs, outputs = info.get("inputs"), info.get("outputs")
    if (
        not isinstance(inputs, list)
        or not isinstance(outputs, list)
        or not 1 <= len(inputs) <= MAX_TEST_CASES
        or len(inputs) != len(outputs)
    ):
        raise ValueError("stdio requires paired nonempty bounded test lists")
    total = 0
    for stdin, expected in zip(inputs, outputs):
        if (
            not isinstance(stdin, str)
            or not isinstance(expected, str)
            or not expected.strip()
        ):
            raise ValueError(
                "stdio tests require string inputs and nonempty expected outputs"
            )
        for value in (stdin, expected):
            size = len(value.encode("utf-8"))
            if size > MAX_CASE_BYTES:
                raise ValueError("stdio testcase exceeds byte limit")
            total += size
    if total > MAX_TEST_BYTES:
        raise ValueError("stdio test suite exceeds byte limit")


def _final_section(text: str) -> str | None:
    # Truncate at the FIRST end marker: text after EOS cannot rescue a response.
    text = _EOS.split(text, maxsplit=1)[0]
    depth = 0
    saw_marker = False
    start = 0
    for marker in re.finditer(r"</?think>", text):
        if marker.group() == "<think>":
            depth += 1
        elif depth:
            depth -= 1
            if not depth:
                start = marker.end()
        elif not saw_marker:
            start = marker.end()  # Template-prefilled opening, or forced close.
        else:
            return None
        saw_marker = True
    return None if depth else text[start:].strip() or None


def _last_boxed(text: str) -> str | None:
    # Same balanced brace walk as eval_hf_math.last_boxed_answer, kept local to
    # avoid importing its model stack and changing USE_HUB_KERNELS at import.
    marker = r"\boxed{"
    for match in reversed(list(re.finditer(re.escape(marker), text))):
        start = match.end()
        depth = 1
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index].strip() or None
        # An unfinished final box must not recover an earlier correct answer.
        return None
    return None


def _answer(final: str) -> str | None:
    # Anchor the field at a line boundary rather than accepting "not Answer:".
    fields = list(
        re.finditer(r"(?im)^[ \t]*(?:\*\*)?Answer[ \t]*:(?:\*\*)?[ \t]*", final)
    )
    if fields:
        # The final field owns its entire tail, including multiline text.
        answer = final[fields[-1].end() :].strip()
        if answer.startswith(r"\boxed{"):
            depth = 1
            for index in range(7, len(answer)):
                depth += (answer[index] == "{") - (answer[index] == "}")
                if depth == 0:
                    if index == len(answer) - 1:
                        return answer[7:index].strip() or None
                    break
        return answer or None
    return _last_boxed(final)


def _stdio_normalize(output: bytes) -> bytes:
    return b"\n".join(
        line.rstrip(b" \t\r") for line in output.replace(b"\r\n", b"\n").split(b"\n")
    ).rstrip(b"\n")


@lru_cache(maxsize=1)
def _seccomp_program() -> bytes:
    """Deny process creation/namespace escape; no source-level Python policy."""
    name = ctypes.util.find_library("seccomp")
    if not name:
        raise RuntimeError("Stdio rewards require libseccomp")
    lib = ctypes.CDLL(name, use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    lib.seccomp_rule_add.restype = ctypes.c_int
    lib.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.seccomp_export_bpf.restype = ctypes.c_int
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    context = lib.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("cannot initialize stdio seccomp filter")
    file = tempfile.TemporaryFile()
    try:
        for syscall in (
            "clone",
            "clone3",
            "fork",
            "vfork",
            "unshare",
            "setns",
            "mount",
            "umount2",
            "pivot_root",
            "chroot",
            "ptrace",
            "process_vm_readv",
            "process_vm_writev",
            "process_madvise",
            "pidfd_getfd",
            "bpf",
            "userfaultfd",
            "keyctl",
            "add_key",
            "request_key",
            "perf_event_open",
            "io_uring_setup",
            "setsid",
            "setpgid",
            "socket",
            "socketpair",
        ):
            number = lib.seccomp_syscall_resolve_name(syscall.encode())
            if (
                number >= 0
                and lib.seccomp_rule_add(context, 0x00050000 | errno.EPERM, number, 0)
                != 0
            ):
                raise RuntimeError(f"cannot restrict syscall {syscall}")
        if lib.seccomp_export_bpf(context, file.fileno()) != 0:
            raise RuntimeError("cannot export stdio seccomp filter")
        file.seek(0)
        return file.read()
    finally:
        file.close()
        lib.seccomp_release(context)


@lru_cache(maxsize=1)
def _runtime_command() -> tuple[str, ...]:
    """Resolve only the system interpreter, stdlib and its shared libraries."""
    bwrap, prlimit, ldd = (shutil.which(name) for name in ("bwrap", "prlimit", "ldd"))
    python = Path("/usr/bin/python3")
    if not bwrap or not prlimit or not ldd or not python.is_file():
        raise RuntimeError(
            "Stdio rewards require bwrap, prlimit, ldd and /usr/bin/python3"
        )
    try:
        probe = subprocess.run(
            [
                str(python),
                "-I",
                "-S",
                "-c",
                "import json,sysconfig;print(json.dumps(sysconfig.get_paths()))",
            ],
            capture_output=True,
            check=True,
            timeout=10,
            env={},
        )
        stdlib = Path(json.loads(probe.stdout)["stdlib"]).resolve()
        extensions = sorted((stdlib / "lib-dynload").glob("*.so"))
        linked = subprocess.run(
            [ldd, str(python.resolve()), *map(str, extensions)],
            capture_output=True,
            check=True,
            timeout=10,
            env={"LC_ALL": "C"},
        ).stdout.decode()
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
        raise RuntimeError("cannot resolve isolated Python runtime") from error
    libraries = set(re.findall(r"(?:=>\s+|^\s+)(/[^\s()]+)\s+\(", linked, re.MULTILINE))
    # ldd may resolve an absolute loader name through another path. The ELF
    # interpreter still needs its original /lib64/... destination mounted.
    libraries.update(re.findall(r"^\s+(/[^\s:]+)\s+=>", linked, re.MULTILINE))
    if not libraries or "not found" in linked:
        raise RuntimeError("cannot resolve Python shared libraries")
    command = [
        prlimit,
        "--cpu=2:2",
        f"--as={384 << 20}:{384 << 20}",
        "--fsize=4096:4096",
        "--nofile=256:256",
        "--core=0:0",
        "--",
        bwrap,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--clearenv",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "LANG",
        "C.UTF-8",
        "--setenv",
        "PYTHONHASHSEED",
        "0",
        "--ro-bind",
        str(python.resolve()),
        "/usr/bin/python3",
        "--ro-bind",
        str(stdlib),
        str(stdlib),
    ]
    # A system site-packages directory is not part of the runtime contract.
    for packages in (stdlib / "site-packages", stdlib / "dist-packages"):
        if packages.exists():
            command.extend(["--tmpfs", str(packages), "--remount-ro", str(packages)])
    for path in sorted(libraries):
        command.extend(["--ro-bind", str(Path(path).resolve()), path])
    command.extend(
        [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--remount-ro",
            "/dev",
            "--dir",
            "/tmp",
            "--chdir",
            "/tmp",
        ]
    )
    return tuple(command)


def _execute_case(code_path: Path, stdin: bytes, seconds: float) -> tuple[str, bytes]:
    """Bounded nonblocking three-pipe pump; always reap the PID namespace."""
    with tempfile.TemporaryFile() as seccomp, tempfile.TemporaryFile() as status:
        seccomp.write(_seccomp_program())
        seccomp.seek(0)
        command = [
            *_runtime_command(),
            "--ro-bind",
            str(code_path),
            "/solution.py",
            "--info-fd",
            str(status.fileno()),
            "--seccomp",
            str(seccomp.fileno()),
            "--remount-ro",
            "/",
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            "/solution.py",
        ]
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={},
                close_fds=True,
                pass_fds=(seccomp.fileno(), status.fileno()),
                start_new_session=True,
            )
        except OSError as error:
            raise RuntimeError("cannot launch stdio sandbox") from error
        output = bytearray()
        errors = bytearray()
        reason = "pass"
        deadline = time.monotonic() + seconds
        try:
            with selectors.DefaultSelector() as selector:
                for pipe in (process.stdin, process.stdout, process.stderr):
                    os.set_blocking(pipe.fileno(), False)
                selector.register(process.stdout, selectors.EVENT_READ, output)
                selector.register(process.stderr, selectors.EVENT_READ, errors)
                offset = 0
                if stdin:
                    selector.register(process.stdin, selectors.EVENT_WRITE, None)
                else:
                    process.stdin.close()
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        reason = "timeout"
                        break
                    for key, _ in selector.select(min(remaining, 0.05)):
                        pipe = key.fileobj
                        if key.events == selectors.EVENT_WRITE:
                            try:
                                offset += os.write(
                                    pipe.fileno(), stdin[offset : offset + 8192]
                                )
                            except BrokenPipeError:
                                offset = len(stdin)
                            if offset == len(stdin):
                                selector.unregister(pipe)
                                pipe.close()
                        else:
                            chunk = os.read(pipe.fileno(), 8192)
                            if not chunk:
                                selector.unregister(pipe)
                                pipe.close()
                            elif len(key.data) + len(chunk) > _OUTPUT_BYTES:
                                reason = "output_limit"
                                break
                            else:
                                key.data.extend(chunk)
                    if reason != "pass":
                        break
                if reason == "pass":
                    try:
                        process.wait(timeout=max(0.001, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        reason = "timeout"
        finally:
            # No candidate can fork, change sessions or process groups. Killing
            # the supervisor additionally destroys the isolated PID namespace.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            for pipe in (process.stdin, process.stdout, process.stderr):
                pipe.close()
        status.seek(0)
        if not status.read(1):
            raise RuntimeError(
                "Stdio sandbox setup failed: " + errors.decode(errors="replace")[:500]
            )
        if reason != "pass":
            return reason, bytes(output)
        if process.returncode != 0:
            return (
                "timeout"
                if process.returncode
                in (
                    -signal.SIGXCPU,
                    -signal.SIGKILL,
                    128 + signal.SIGXCPU,
                    128 + signal.SIGKILL,
                )
                else "runtime_error"
            ), bytes(output)
        return "pass", bytes(output)


@lru_cache(maxsize=1)
def _check_runtime() -> None:
    """Infrastructure failures must raise, never poison rewards as wrong code."""
    _runtime_command()
    with tempfile.TemporaryDirectory(prefix="verifier-runtime-") as directory:
        path = Path(directory) / "check.py"
        path.write_text(
            "import sys, math, hashlib\nprint('runtime-ready')\n", encoding="utf-8"
        )
        reason, output = _execute_case(path, b"", _CASE_WALL_SECONDS)
    if reason != "pass" or output != b"runtime-ready\n":
        raise RuntimeError(f"isolated Python runtime preflight failed: {reason}")


def _score_code(code: str, info: dict) -> tuple[bool, str]:
    if len(code.encode("utf-8")) > MAX_CODE_BYTES:
        return False, "code_limit"
    with (
        _SANDBOX_SLOTS,
        tempfile.TemporaryDirectory(prefix="verifiable-stdio-") as directory,
    ):
        path = Path(directory) / "solution.py"
        path.write_text(code, encoding="utf-8")
        path.chmod(0o400)
        # Resolve runtime outside the candidate's cumulative wall allowance.
        _check_runtime()
        deadline = time.monotonic() + _TOTAL_WALL_SECONDS
        for stdin, expected in zip(info["inputs"], info["outputs"]):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False, "timeout"
            reason, actual = _execute_case(
                path, stdin.encode("utf-8"), min(_CASE_WALL_SECONDS, remaining)
            )
            if reason != "pass":
                return False, reason
            if _stdio_normalize(actual) != _stdio_normalize(expected.encode("utf-8")):
                return False, "wrong_output"
    return True, "pass"


def score_verifiable_response(row: dict, text: str) -> tuple[bool, str]:
    validate_verifiable_task(row)
    final = _final_section(text) if isinstance(text, str) else None
    if final is None:
        return False, "missing_final"
    kind = row["verification_info"]["kind"]
    if kind == "python_stdio":
        code = extract_code(final)
        return (
            (False, "format_ineligible")
            if code is None
            else _score_code(code, row["verification_info"])
        )
    prediction = _answer(final)
    if prediction is None:
        return False, "missing_answer"
    truth = row["reward_model"]["ground_truth"]
    if kind == "math":
        target = core.parse_numeric_answer(truth)
        if target is not None:
            correct = core.parse_numeric_answer(prediction) == target
        else:
            normalized = core.normalize_final_answer(prediction)
            correct = bool(normalized) and normalized == core.normalize_final_answer(
                truth
            )
    else:
        correct = " ".join(prediction.split()) == " ".join(truth.split())
    return correct, "pass" if correct else "wrong_answer"


@lru_cache(maxsize=1)
def verifiable_reward_identity() -> str:
    """Bind all grading/extraction code and Minerva tables to resume identity."""
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for function in (
        extract_code,
        core.normalize_final_answer,
        core.parse_numeric_answer,
    ):
        digest.update(inspect.getsource(function).encode())
    digest.update(
        json.dumps(
            [core.SUBSTITUTIONS, core.REMOVED_EXPRESSIONS], ensure_ascii=False
        ).encode()
    )
    return f"{VERIFIABLE_TASK_SCHEMA}:{digest.hexdigest()}"
