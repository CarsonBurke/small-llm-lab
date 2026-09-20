"""Behavioral contracts for final answers and the real isolated stdio grader."""

import ctypes.util
import hashlib
import shutil

import pytest

from postraining.verifiable_tasks import (
    MAX_CASE_BYTES,
    MAX_TEST_CASES,
    VERIFIABLE_TASK_SCHEMA,
    is_verifiable_task,
    score_verifiable_response,
    validate_verifiable_task,
)


def task(kind="math", truth="42", *, inputs=None, outputs=None):
    info = {"schema": VERIFIABLE_TASK_SCHEMA, "kind": kind}
    if kind == "python_stdio":
        info.update(
            call_type="std",
            fn_name=None,
            inputs=["2 3\n", "-8 2\n"] if inputs is None else inputs,
            outputs=["5\n", "-6\n"] if outputs is None else outputs,
        )
    return {
        "prompt": [{"role": "user", "content": "Solve the task."}],
        "data_source": "any-dataset",
        "reward_model": {"style": VERIFIABLE_TASK_SCHEMA, "ground_truth": truth},
        "extra_info": {"domain": "arbitrary-reporting-label"},
        "verification_info": info,
    }


@pytest.mark.parametrize(
    "text",
    [
        "<think>Answer: 42",
        "<think>\\boxed{42}",
        "<think>Answer: 42</think>Answer: 0",
        "Answer: 0<|im_end|>Answer: 42",
        "<think>Answer: 42<|im_end|></think>Answer: 42",
        "The scratch calculation gave 42.",
        "not Answer: 42",
        "\\boxed{42} but then \\boxed{",
    ],
)
def test_scratchpad_or_unfinished_answer_cannot_earn_reward(text):
    assert score_verifiable_response(task(), text)[0] is False


@pytest.mark.parametrize(
    "text",
    [
        "<think>Answer: 0</think>Answer: 42<|im_end|>",
        "earlier reasoning</think>Answer: 42</s>",
        "<think>truncated reasoning\n</think>\\boxed{42}",
    ],
)
def test_completed_prefilled_and_forced_closing_thinking(text):
    assert score_verifiable_response(task(), text) == (True, "pass")


def test_math_numeric_equivalence_and_candidate_list_guard():
    assert score_verifiable_response(task(truth="0.5"), r"Answer: \frac{1}{2}")[0]
    assert not score_verifiable_response(task(truth="34"), "Answer: 3, 4")[0]
    assert not score_verifiable_response(task(truth="42"), "Answer: 42\nor perhaps 41")[
        0
    ]
    assert score_verifiable_response(
        task(truth=r"\frac{x}{y}"), r"Final: \boxed{\frac{x}{y}}"
    )[0]
    assert not score_verifiable_response(
        task(truth=r"\frac{x}{y}"), r"\boxed{\frac{y}{x}}"
    )[0]


def test_text_preserves_meaning_and_dispatch_ignores_domain():
    row = task("text", "The United States")
    row["extra_info"]["domain"] = "Math"
    assert score_verifiable_response(row, "Answer: The  United\n States") == (
        True,
        "pass",
    )
    assert not score_verifiable_response(row, "Answer: United States")[0]
    assert not score_verifiable_response(task("text", "A+B"), "Answer: A-B")[0]
    assert not score_verifiable_response(task("text", "a b"), "Answer: ab")[0]
    assert not score_verifiable_response(task("text", "Co"), "Answer: CO")[0]


def test_boxed_answer_field_preserves_text_and_rejects_extra_candidates():
    row = task("text", "Co")
    assert score_verifiable_response(row, r"Answer: \boxed{Co}") == (True, "pass")
    assert not score_verifiable_response(row, r"Answer: \boxed{CO}")[0]
    assert not score_verifiable_response(row, r"Answer: \boxed{Co} or Ni")[0]
    assert not score_verifiable_response(row, r"Answer: \boxed{Ni} \boxed{Co}")[0]
    assert score_verifiable_response(
        task(truth=r"\frac{x}{y}"), r"Answer: \boxed{\frac{x}{y}}"
    )[0]


@pytest.mark.parametrize(
    "change",
    [
        {"verification_info": None},
        {"verification_info": {"schema": VERIFIABLE_TASK_SCHEMA, "kind": "unknown"}},
        {"reward_model": {"style": VERIFIABLE_TASK_SCHEMA, "ground_truth": []}},
        {"prompt": []},
        {"extra_info": {"domain": ""}},
    ],
)
def test_malformed_generic_tasks_fail_closed(change):
    row = task()
    row.update(change)
    with pytest.raises(ValueError):
        validate_verifiable_task(row)


def test_schema_on_either_side_selects_validation_not_legacy_fallback():
    row = task()
    row["reward_model"]["style"] = "invalid"
    assert is_verifiable_task(row)
    with pytest.raises(ValueError):
        score_verifiable_response(row, "Answer: 42")
    assert not is_verifiable_task({"data_source": "ultradata_rl"})
    future = task()
    future["reward_model"]["style"] = "verifiable_task/v999"
    future["verification_info"] = None
    assert is_verifiable_task(future)
    with pytest.raises(ValueError):
        validate_verifiable_task(future)


def test_arrow_nullable_code_fields_do_not_turn_text_into_code():
    row = task("text", "Co")
    row["verification_info"].update(
        call_type=None, fn_name=None, inputs=None, outputs=None
    )
    assert score_verifiable_response(row, "Answer: Co") == (True, "pass")


@pytest.mark.parametrize(
    "inputs,outputs",
    [
        ([], []),
        ([""], []),
        ([""], [""]),
        ([""], [" \n"]),
        ([1], ["1"]),
        ([""], [1]),
        (["x"] * (MAX_TEST_CASES + 1), ["x"] * (MAX_TEST_CASES + 1)),
        (["x" * (MAX_CASE_BYTES + 1)], ["x"]),
    ],
)
def test_missing_or_malformed_tests_are_errors_not_vacuous_success(inputs, outputs):
    with pytest.raises(ValueError):
        validate_verifiable_task(task("python_stdio", inputs=inputs, outputs=outputs))


sandbox = pytest.mark.skipif(
    any(shutil.which(name) is None for name in ("bwrap", "prlimit", "ldd"))
    or ctypes.util.find_library("seccomp") is None,
    reason="real stdio sandbox dependencies unavailable",
)


@sandbox
def test_real_stdio_runs_every_case_and_allows_standard_library():
    row = task("python_stdio")
    code = "import sys, functools, operator\nprint(functools.reduce(operator.add, map(int, sys.stdin.read().split())))"
    assert score_verifiable_response(row, f"```python\n{code}\n```<|im_end|>") == (
        True,
        "pass",
    )
    assert score_verifiable_response(row, "print(5)") == (False, "wrong_output")


@sandbox
@pytest.mark.parametrize(
    "code,reason",
    [
        ("pass", "wrong_output"),
        ("import sys\nprint(0)\nsys.exit(0)", "wrong_output"),
        ("raise RuntimeError('bad program')", "runtime_error"),
        ("while True: pass", "timeout"),
        ("import os\nwhile True: os.write(1, b'x' * 8192)", "output_limit"),
        ("import os\nwhile True: os.write(2, b'x' * 8192)", "output_limit"),
    ],
)
def test_runtime_failures_and_exit_zero_do_not_forge_success(code, reason):
    assert score_verifiable_response(task("python_stdio"), code) == (False, reason)


@sandbox
def test_final_code_is_not_recovered_from_scratchpad_or_unclosed_fence():
    correct = "```python\nprint(sum(map(int,input().split())))\n```"
    row = task("python_stdio")
    assert not score_verifiable_response(row, "<think>" + correct)[0]
    assert score_verifiable_response(
        row, "<think>" + correct + "</think>```python\nprint(0)\n```"
    ) == (False, "wrong_output")
    assert score_verifiable_response(row, correct + "\n```python\nprint(0)") == (
        False,
        "format_ineligible",
    )
    assert score_verifiable_response(row, "reasoning</think>" + correct) == (
        True,
        "pass",
    )


@sandbox
def test_output_normalization_preserves_leading_and_internal_whitespace():
    row = task("python_stdio", inputs=[""], outputs=["a b\nc\n"])
    assert score_verifiable_response(row, "print('a b  \\r\\nc\\t\\n')")[0]
    assert not score_verifiable_response(row, "print(' a b\\nc')")[0]
    assert not score_verifiable_response(row, "print('a  b\\nc')")[0]


@sandbox
def test_hidden_expected_outputs_and_host_secrets_are_not_exposed(
    tmp_path, monkeypatch
):
    secret = tmp_path / "host-secret"
    secret.write_text("not-visible")
    monkeypatch.setenv("VERIFIER_TEST_SECRET", "not-visible")
    expected = hashlib.sha256(b"input-value").hexdigest()
    row = task("python_stdio", inputs=["input-value"], outputs=[expected])
    code = (
        "import os, sys, hashlib\n"
        "assert 'VERIFIER_TEST_SECRET' not in os.environ\n"
        f"assert not os.path.exists({str(secret)!r})\n"
        "assert not os.path.exists('/etc/passwd')\n"
        "assert not os.path.exists('/home')\n"
        "assert set(os.listdir('/tmp')) == set()\n"
        "assert sys.argv == ['/solution.py']\n"
        "for fd in range(3, 32):\n"
        "    try: os.fstat(fd)\n"
        "    except OSError: pass\n"
        "    else: raise AssertionError('unexpected inherited descriptor')\n"
        "print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())\n"
    )
    assert score_verifiable_response(row, code) == (True, "pass")


@sandbox
def test_network_fork_and_writable_candidate_are_denied():
    row = task("python_stdio", inputs=[""], outputs=["isolated"])
    code = (
        "import os, socket\n"
        "for operation in (os.fork, socket.socket, lambda: open('/solution.py','w'), lambda: open('/tmp/file','w')):\n"
        "    try: operation()\n"
        "    except OSError: pass\n"
        "    else: raise AssertionError('sandbox escape')\n"
        "print('isolated')\n"
    )
    assert score_verifiable_response(row, code) == (True, "pass")


@sandbox
def test_full_input_and_output_pipes_do_not_deadlock():
    row = task("python_stdio", inputs=["x" * (128 << 10)], outputs=["y" * (128 << 10)])
    code = "import sys\nsys.stdout.write('y'*(128<<10)); sys.stdout.flush()\nsys.stdin.read()"
    assert score_verifiable_response(row, code) == (True, "pass")
