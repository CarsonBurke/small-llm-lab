import shutil
import subprocess

import pytest

from postraining.vapo import code_reward
from postraining.vapo.code_reward import (
    PYTHON_REWARD_SCHEMA, batch_python_test_scores, python_test_result, python_test_score,
)

pytestmark = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap unavailable")


def verification(tests, setup=()):
    return {"schema": PYTHON_REWARD_SCHEMA, "entry_points": ["f"],
            "test_setup": list(setup), "tests": tests}


def test_fraction_counts_later_passes_after_assertion_and_runtime_failures():
    info = verification(["assert f(0) == 9", "assert f(1) == 1", "assert f(2) == 2", "assert f(3) == 3"])
    candidate = "def f(x):\n    if x == 2: raise ValueError('bad input')\n    return x"
    score = python_test_score(candidate, info)
    assert (score.passed, score.total, score.status, score.reward) == (2, 4, "tests_failed", .5)
    assert python_test_result(candidate, info) == "tests_failed"


def test_setup_does_not_inflate_denominator_and_binary_fixtures_remain_supported():
    code = "def f(x): return x + 1"
    info = verification(["assert f(value) == 3"], ["value = 2"])
    score = python_test_score(code, info)
    assert (score.passed, score.total, score.reward) == (1, 1, 1.0)
    legacy = verification(["value = 2", "assert f(value) == 3"])
    assert python_test_result(code, legacy) == "pass"
    grouped = python_test_score(code, legacy)
    assert (grouped.passed, grouped.total, grouped.reward) == (1, 1, 1.0)
    with pytest.raises(ValueError, match="assertion"):
        python_test_score(code, verification(["value = 2"]))


def test_grouped_fixture_failure_does_not_skip_later_independent_cases():
    info = verification(["value = f(0)", "assert value == 0", "value = f(2)", "assert value == 2", "unscored = 999"])
    code = "def f(x):\n    if x == 0: raise ValueError('bad')\n    return x"
    score = python_test_score(code, info)
    assert (score.passed, score.total, score.reward) == (1, 2, .5)


def test_compound_assertion_blocks_have_fixed_case_denominator():
    info = verification(["for x in range(4):\n    assert f(x) == x", "assert f(9) == 9"])
    score = python_test_score("def f(x): return 9", info)
    assert (score.passed, score.total, score.reward) == (1, 2, .5)


def test_unused_helper_assertions_never_earn_credit():
    info = verification(["def unused_check():\n    assert False", "assert f(1)==1"])
    score = python_test_score("def f(x): return 0", info)
    assert (score.passed, score.total, score.reward) == (0, 1, 0.0)


def test_lazy_generator_helper_does_not_manufacture_a_case():
    info = verification(["def lazy_check():\n    yield 0\n    assert False", "lazy_check()", "assert f(1)==1"])
    score = python_test_score("def f(x): return 0", info)
    assert (score.passed, score.total, score.reward) == (0, 1, 0.0)


def test_helper_calls_count_independent_cases_including_trailing_calls():
    info = verification([
        "def check(x):\n    assert f(x)==x",
        "def wrapper(x):\n    check(x)",
        "wrapper(1)", "wrapper(2)", "wrapper(3)",
    ])
    score = python_test_score("def f(x): return 2", info)
    assert (score.passed, score.total, score.reward) == (1, 3, 1/3)


def test_helper_call_assignments_remain_fixtures_for_next_assertion():
    info = verification([
        "def check(x):\n    assert f(x)==x\n    return x",
        "value = check(1)", "assert value==1", "check(2)",
    ])
    score = python_test_score("def f(x): return 2", info)
    assert (score.passed, score.total, score.reward) == (1, 2, .5)


@pytest.mark.parametrize("candidate,status", [
    ("import os\nos._exit(0)", "policy_rejected"),
    ("def f(:", "policy_rejected"),
    ("raise ValueError('initialization failed')", "tests_failed"),
    ("def other(x): return x", "tests_failed"),
])
def test_invalid_candidates_have_zero_credit_with_full_denominator(candidate, status):
    score = python_test_score(candidate, verification(["assert f(1) == 1", "assert f(2) == 2"]))
    assert (score.passed, score.total, score.status, score.reward) == (0, 2, status, 0.0)


def test_mutating_library_class_invalidates_earlier_passes():
    code = "from collections import Counter\nC = Counter\ndef f(x):\n    if x == 2: C.most_common = lambda self, n: []\n    return x"
    assert code_reward.python_candidate_allowed(code)
    score = python_test_score(code, verification(["assert f(1) == 1", "assert f(2) == 2"]))
    assert (score.passed, score.total, score.reward) == (0, 2, 0.0)


def test_fixture_mutation_is_checked_before_following_assertion():
    # Grouping fixtures must not weaken v6's per-original-statement checks.
    # This candidate restores the class on the next call, hiding its patch
    # if integrity were checked only after the complete fixture/assert case.
    code = "from collections import Counter\nC = Counter\nsaved = C.most_common\ndef f(x):\n    if x == 1: C.most_common = lambda self, n: []\n    else: C.most_common = saved\n    return x"
    info = verification(["assert f(0)==0", "value = f(1)", "assert f(2)==2"])
    score = python_test_score(code, info)
    assert (score.passed, score.total, score.reward) == (0, 2, 0.0)


def test_candidate_cannot_forge_counts_or_test_namespace():
    code = "_harness_passed = 999\n_harness_test_ok = True\nprint('999:999')\nabs = lambda x: 0\ndef f(x): return x"
    score = python_test_score(code, verification(["assert abs(f(1)) == 0", "assert f(2) == 2"]))
    assert (score.passed, score.total, score.reward) == (1, 2, .5)


def test_timeout_never_returns_credit_for_prior_passes(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"], output=b"1")
    monkeypatch.setattr(code_reward.subprocess, "run", timeout)
    score = python_test_score("def f(x): return x", verification(["assert f(1)==1", "assert f(2)==2"]))
    assert (score.passed, score.total, score.status, score.reward) == (0, 2, "timeout", 0.0)


def test_batch_preserves_order_and_handles_markdown_and_empty_batches():
    info = verification(["assert f(1)==1", "assert f(2)==2"])
    scores = batch_python_test_scores(["```python\ndef f(x): return x\n```", "def f(x): return 1", "def f(x): return 0"], info, workers=2)
    assert [score.reward for score in scores] == [1.0, .5, 0.0]
    assert batch_python_test_scores([], info) == []
