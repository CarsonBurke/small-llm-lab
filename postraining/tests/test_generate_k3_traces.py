from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from postraining.generate_k3_traces import (
    DAPO_HEADER,
    ContractError,
    CostMeter,
    TeacherClient,
    YieldCanary,
    bare_problem,
    effort_body,
    finished_keys,
    interleave_by_source,
    run_problem,
    split_visible_solution,
)
from postraining.math_prompt import (
    CHINESE_ANSWER_FIELD_INSTRUCTION,
    ANSWER_FIELD_INSTRUCTIONS,
)


def test_bare_problem_strips_all_known_templates():
    # gsm8k layout: problem first, reminder sentence after.
    gsm8k_row = {
        "prompt": [
            {
                "content": (
                    "Natalia sold clips to 48 friends. How many?\n\n"
                    f"{ANSWER_FIELD_INSTRUCTIONS[1]}"
                )
            }
        ]
    }
    assert bare_problem(gsm8k_row) == "Natalia sold clips to 48 friends. How many?"

    # deepmind layout: header + field sentence first, problem after.
    deepmind_row = {
        "prompt": [
            {
                "content": (
                    f"{DAPO_HEADER} {ANSWER_FIELD_INSTRUCTIONS[0]}\n\n"
                    "What is -304863 less than 0.33735?"
                )
            }
        ]
    }
    assert bare_problem(deepmind_row) == "What is -304863 less than 0.33735?"

    # The Chinese dapo template is in the authoritative rewrite table and
    # must strip too (red-team finding 9).
    chinese = CHINESE_ANSWER_FIELD_INSTRUCTION
    row = {
        "prompt": [
            {"content": f"{ANSWER_FIELD_INSTRUCTIONS[0]}\n\n某题。\n{chinese}"}
        ]
    }
    assert bare_problem(row) == "某题。"

    # Fail-loud: an unknown Answer: template must never reach the teacher.
    with pytest.raises(ValueError, match="left an Answer: demand"):
        bare_problem(
            {"prompt": [{"content": "Q?\n\nReply with Answer: \\boxed{x}."}]}
        )
    with pytest.raises(ValueError, match="empty problem"):
        bare_problem({"prompt": [{"content": f"  {DAPO_HEADER}  "}]})


def test_bare_problem_strips_real_parquet_rows():
    from postraining.core import load_unique_math_rows
    from postraining.math_prompt import ANSWER_FIELD_DEMAND

    data_dir = Path(__file__).resolve().parents[1] / "data"
    parquets = [
        path
        for path in (
            data_dir / "gsm8k_rl_prompts.parquet",
            data_dir / "deepmind-interpolate-rl.parquet",
            data_dir / "deepmind-interpolate-easy.parquet",
        )
        if path.exists()
    ]
    if not parquets:
        pytest.skip("no RL prompt parquets present")
    for path in parquets:
        for row in load_unique_math_rows(str(path))[:50]:
            problem = bare_problem(row)
            assert problem
            assert not ANSWER_FIELD_DEMAND.search(problem)
            assert DAPO_HEADER not in problem


def test_split_visible_solution_contract():
    split = split_visible_solution("First add.\nThen halve.\nAnswer: 72\n")
    assert split == ("First add.\nThen halve.", "72")
    # Case-insensitive, matching the verifier's field pattern.
    assert split_visible_solution("reason\nanswer: 5") == ("reason", "5")
    # Violations: no answer line, empty value, prose after the answer.
    assert split_visible_solution("just reasoning") is None
    assert split_visible_solution("reason\nAnswer:") is None
    assert split_visible_solution("reason\nAnswer: 72\ntrailing chat") is None
    assert split_visible_solution("") is None
    # Answer must OPEN the line: a mid-line mention is not the field.
    assert split_visible_solution("reason\nso the Answer: 72") is None


def test_effort_body_shapes():
    assert effort_body("none", "reasoning_effort") == {}
    assert effort_body("low", "reasoning_effort") == {"reasoning_effort": "low"}
    # Dotted keys nest for providers like OpenRouter.
    assert effort_body("high", "reasoning.effort") == {
        "reasoning": {"effort": "high"}
    }


def test_interleave_by_source_round_robins():
    problems = [
        {"key": f"gsm8k/{i}"} for i in range(4)
    ] + [{"key": f"deepmind/{i}"} for i in range(2)]
    order = [p["key"] for p in interleave_by_source(problems)]
    # Any prefix samples both sources: a --limit 2 pilot sees one of each.
    assert order == [
        "gsm8k/0", "deepmind/0", "gsm8k/1", "deepmind/1", "gsm8k/2", "gsm8k/3",
    ]


def test_finished_keys_resume_semantics(tmp_path):
    output = tmp_path / "traces.jsonl"
    assert finished_keys(output) == set()
    records = [
        {"key": "gsm8k/0", "correct": True},
        {"key": "gsm8k/1", "correct": False},  # failed attempt: retry
        {"key": "gsm8k/2", "style": "minerva", "exhausted": "schedule"},
        {"key": "gsm8k/1", "correct": False},
    ]
    text = "".join(json.dumps(r) + "\n" for r in records)
    # A torn trailing line (writer killed mid-record) must not kill
    # resume — its problem simply retries (red-team finding 10).
    output.write_text(text + '{"key": "gsm8k/3", "corr')
    assert finished_keys(output) == {"gsm8k/0", "gsm8k/2"}
    # After fixing a broken format contract, the problems whose failures
    # detected it must be regenerable (red-team round 2, NEW-1).
    assert finished_keys(output, retry_exhausted=True) == {"gsm8k/0"}


def test_cost_meter_budget_ceiling():
    meter = CostMeter(budget_usd=1.0, price_in=2.0, price_out=10.0)
    meter.charge({"prompt_tokens": 100_000, "completion_tokens": 50_000})
    # 0.1M * $2/M + 0.05M * $10/M = 0.2 + 0.5 = 0.7
    assert meter.spent_usd() == pytest.approx(0.7)
    assert not meter.exhausted()
    meter.charge({"prompt_tokens": 0, "completion_tokens": 30_000})
    assert meter.spent_usd() == pytest.approx(1.0)
    assert meter.exhausted()


def _client(meter, post):
    client = TeacherClient(
        "https://example.invalid/v1", "key", "kimi-k3", 1.0, meter,
        max_retries=3,
    )
    client._post = post
    return client


def test_missing_usage_is_fatal_and_charged():
    """A usage-less 200 must charge an estimate and stop the run.

    Red-team finding 1: a silent zero would disarm the budget ceiling —
    the meter would read $0 while the provider bills every request.
    """
    meter = CostMeter(budget_usd=1.0, price_in=2.0, price_out=10.0)
    body = {"choices": [{"message": {"content": "Answer: 2"}}], "usage": {}}
    client = _client(meter, lambda payload: (200, "", body))
    with pytest.raises(ContractError, match="no usable usage"):
        client.complete("sys", "1+1?", 0.6, 2048, {})
    assert meter.spent_usd() > 0.0
    # Null (or junk) usage values must reach the same charged fail-closed
    # branch, not crash uncharged (red-team round 2, NEW-4).
    null_meter = CostMeter(budget_usd=1.0, price_in=2.0, price_out=10.0)
    null_body = {
        "choices": [{"message": {"content": "Answer: 2"}}],
        "usage": {"prompt_tokens": None, "completion_tokens": None},
    }
    client = _client(null_meter, lambda payload: (200, "", null_body))
    with pytest.raises(ContractError, match="no usable usage"):
        client.complete("sys", "1+1?", 0.6, 2048, {})
    assert null_meter.spent_usd() > 0.0


def test_timeout_retries_charge_estimate_but_429_does_not():
    """Red-team finding 2: the provider may bill attempts we never see."""
    meter = CostMeter(budget_usd=100.0, price_in=2.0, price_out=10.0)

    def timeout_post(payload):
        raise requests.exceptions.Timeout("deadline")

    client = _client(meter, timeout_post)
    with pytest.raises(RuntimeError, match="failed after retries"):
        client.complete("sys", "1+1?", 0.6, 2048, {})
    # Every timed-out attempt was charged conservatively.
    assert meter.completion_tokens == 3 * 2048

    rejected = CostMeter(budget_usd=100.0, price_in=2.0, price_out=10.0)
    client = _client(rejected, lambda payload: (429, "slow down", None))
    with pytest.raises(RuntimeError, match="failed after retries"):
        client.complete("sys", "1+1?", 0.6, 2048, {})
    # Rejected requests were never processed, so nothing is charged.
    assert rejected.spent_usd() == 0.0


def test_successful_completion_charges_real_usage():
    meter = CostMeter(budget_usd=100.0, price_in=2.0, price_out=10.0)
    body = {
        "choices": [
            {
                "message": {
                    "content": "One plus one is two.\nAnswer: 2",
                    "reasoning_content": "1+1=2",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 40, "completion_tokens": 30},
    }
    client = _client(meter, lambda payload: (200, "", body))
    result = client.complete("sys", "1+1?", 0.6, 2048, {})
    assert result["visible"].endswith("Answer: 2")
    assert result["reasoning"] == "1+1=2"
    assert meter.prompt_tokens == 40 and meter.completion_tokens == 30


class _ScriptedClient:
    """run_problem test double: pops one scripted response per call."""

    model = "kimi-k3"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system, problem, temperature, max_tokens, extra):
        self.calls.append({"system": system, "max_tokens": max_tokens,
                           "extra": extra})
        return self.responses.pop(0)


def _run_args(**overrides):
    defaults = dict(
        temperature=0.6, max_tokens=64, max_tokens_cap=256,
        effort_key="reasoning_effort",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _response(visible, finish_reason="stop"):
    return {
        "visible": visible,
        "reasoning": "",
        "finish_reason": finish_reason,
        "usage": {"prompt_tokens": 10, "completion_tokens": 10},
    }


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    from postraining.core import GPT2BPETokenizer

    return GPT2BPETokenizer()


def test_run_problem_truncation_doubles_tokens_not_effort(gpt2_tokenizer):
    """Red-team finding 6: escalating effort on truncation compounds it."""
    meter = CostMeter(budget_usd=100.0, price_in=2.0, price_out=10.0)
    canary = YieldCanary(min_resolved=1000, min_yield=0.2)
    client = _ScriptedClient([
        _response("cut off mid-thou", finish_reason="length"),
        _response("cut again", finish_reason="length"),
        _response("Add them.\nAnswer: 72"),
    ])
    records = []
    problem = {"key": "gsm8k/0", "problem": "sum?", "ground_truth": "72",
               "style": "minerva"}
    run_problem(
        problem, client, ["low", "high"], _run_args(), meter,
        gpt2_tokenizer, records.append, canary,
    )
    # Same effort slot throughout; only the token cap doubled (64->128->256).
    assert [c["extra"] for c in client.calls] == [
        {"reasoning_effort": "low"}] * 3
    assert [c["max_tokens"] for c in client.calls] == [64, 128, 256]
    assert records[-1]["correct"] is True and records[-1]["attempt"] == 0


def test_run_problem_canary_halts_style(gpt2_tokenizer):
    """Red-team finding 3: deterministic format mismatch must halt early."""
    meter = CostMeter(budget_usd=100.0, price_in=2.0, price_out=10.0)
    canary = YieldCanary(min_resolved=2, min_yield=0.5)
    args = _run_args()
    records = []
    # Two problems exhaust their schedule with wrong canonical form.
    for index in range(2):
        client = _ScriptedClient([_response("So it is 0.6.\nAnswer: 0.6")])
        run_problem(
            {"key": f"deepmind/{index}", "problem": "?", "ground_truth": "3/5",
             "style": "exact"},
            client, ["low"], args, meter, gpt2_tokenizer, records.append,
            canary,
        )
    assert canary.aborted("exact")
    # The exhausted records are tagged for targeted --retry-exhausted
    # reruns (red-team round 2, NEW-1).
    exhausted = [r for r in records if r.get("exhausted")]
    assert all(r["exhausted"] == "schedule" for r in exhausted)
    assert all(r["style"] == "exact" for r in exhausted)
    # A third exact-style problem is skipped without any request or record.
    client = _ScriptedClient([])
    before = len(records)
    run_problem(
        {"key": "deepmind/9", "problem": "?", "ground_truth": "1/2",
         "style": "exact"},
        client, ["low"], args, meter, gpt2_tokenizer, records.append, canary,
    )
    assert client.calls == [] and len(records) == before
    # Other styles keep running.
    assert not canary.aborted("minerva")


def test_canary_zero_yield_fires_at_half_sample():
    """Red-team round 2, NEW-2: a deterministic contract failure yields
    exactly zero hits, so detection fires at half the sample cost."""
    canary = YieldCanary(min_resolved=10, min_yield=0.2)
    for _ in range(5):
        canary.resolve("exact", False)
    assert canary.aborted("exact")
    assert canary.aborted_styles() == {"exact"}
    # A single early hit disarms the zero-yield trigger: the style then
    # needs the full sample before the fractional floor can judge it.
    mixed = YieldCanary(min_resolved=10, min_yield=0.2)
    mixed.resolve("exact", True)
    for _ in range(8):
        mixed.resolve("exact", False)
    assert not mixed.aborted("exact")
    mixed.resolve("exact", False)  # 1/10 < 0.2 at the full sample
    assert mixed.aborted("exact")


def test_run_problem_exact_style_gets_format_hint(gpt2_tokenizer):
    from postraining.generate_k3_traces import EXACT_STYLE_HINT, SYSTEM_PROMPT

    meter = CostMeter(budget_usd=100.0, price_in=2.0, price_out=10.0)
    canary = YieldCanary(min_resolved=1000, min_yield=0.2)
    client = _ScriptedClient([_response("Lowest terms.\nAnswer: 3/5")])
    run_problem(
        {"key": "deepmind/0", "problem": "?", "ground_truth": "3/5",
         "style": "exact"},
        client, ["low"], _run_args(), meter, gpt2_tokenizer,
        lambda record: None, canary,
    )
    assert client.calls[0]["system"] == SYSTEM_PROMPT + EXACT_STYLE_HINT
    # The hint is value-blind: nothing from the ground truth reaches it.
    assert "3/5" not in EXACT_STYLE_HINT.replace("like 3/5", "")

    client = _ScriptedClient([_response("Add.\nAnswer: 72")])
    run_problem(
        {"key": "gsm8k/0", "problem": "?", "ground_truth": "72",
         "style": "minerva"},
        client, ["low"], _run_args(), meter, gpt2_tokenizer,
        lambda record: None, canary,
    )
    assert client.calls[0]["system"] == SYSTEM_PROMPT


def test_empty_effort_schedule_is_rejected(monkeypatch, tmp_path):
    """Red-team finding 5: an empty schedule would mark every problem
    exhausted with zero requests, permanently poisoning resume."""
    from postraining import generate_k3_traces

    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_k3_traces",
            "--output", str(tmp_path / "out.jsonl"),
            "--gsm8k-rows", "1",
            "--budget-usd", "1",
            "--price-in-per-mtok", "1",
            "--price-out-per-mtok", "1",
            "--effort-schedule", " , ",
        ],
    )
    with pytest.raises(SystemExit):
        generate_k3_traces.main()
