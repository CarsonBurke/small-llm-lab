from __future__ import annotations

import json

import pytest

from postraining.benchmark_report import (
    BENCHMARK_ANSWER_SCHEMA,
    render_benchmark_report,
    sample_json_to_report_payload,
    write_sample_json_report,
    write_benchmark_report,
)


def _attempt(problem: int, sample: int) -> dict[str, object]:
    forced = sample % 2 == 0
    return {
        "problem_index": problem,
        "dataset_index": f"dataset-{problem}",
        "sample_index": sample,
        "prompt": "Find <x> & explain",
        "ground_truth": "<42>",
        "answer_style": "exact",
        "emitted_text": "<script>alert('no')</script>\nAnswer: 42",
        "parsed_answer": "42",
        "correct": sample == 0,
        "terminated": sample != 3,
        "termination_token_id": 2 if sample != 3 else None,
        "emitted_token_count": 8,
        "forced_initial_think": forced,
        "forced_thought_count": int(forced),
        "optional_thought_count": 2,
        "total_thought_count": 2 + int(forced),
        "think_run_lengths": [1, 1] if not forced else [1, 2],
        "action_trace": "TETEE" if not forced else "TETTE",
    }


def _payload() -> dict[str, object]:
    return {
        "schema": BENCHMARK_ANSWER_SCHEMA,
        "step": 80,
        "reward_schema": "test_reward/v1",
        "metrics": {
            "accuracy": 0.25,
            "forced_initial_accuracy": 0.125,
            "unforced_initial_accuracy": 0.375,
            "think_fraction": 0.2,
            "samples": 1152,
        },
        "attempts": [
            _attempt(problem, sample)
            for problem in reversed(range(4))
            for sample in reversed(range(4))
        ],
    }


def test_render_benchmark_report_is_self_contained_and_escapes_model_text():
    report = render_benchmark_report(_payload())

    assert "What the model answered" in report
    assert "Overall accuracy</span><strong>25.00%" in report
    assert "test_reward/v1" in report
    assert "exact" in report
    assert report.count('class="attempt ') == 16
    assert "&lt;script&gt;alert" in report
    assert "<script>" not in report
    assert "https://" not in report
    assert "src=" not in report


def test_write_benchmark_report_keeps_history_and_atomically_updates_latest(tmp_path):
    payload = _payload()
    paths = write_benchmark_report(
        tmp_path,
        payload["step"],
        payload["metrics"],
        payload["attempts"],
    )

    assert paths["history"] == tmp_path / "bench_answers" / "step_000080.json"
    assert paths["latest"] == tmp_path / "bench_answers" / "latest.json"
    assert paths["report"] == tmp_path / "bench_answers.html"
    machine_readable = json.loads(paths["latest"].read_text())
    assert machine_readable["schema"] == BENCHMARK_ANSWER_SCHEMA
    assert len(machine_readable["attempts"]) == 16
    assert [
        (attempt["problem_index"], attempt["sample_index"])
        for attempt in machine_readable["attempts"]
    ] == [(problem, sample) for problem in range(4) for sample in range(4)]
    assert paths["history"].read_text() == paths["latest"].read_text()
    assert not list(tmp_path.rglob("*.tmp"))


def test_benchmark_report_rejects_incomplete_capture():
    payload = _payload()
    with pytest.raises(ValueError, match="exactly the first four"):
        render_benchmark_report({**payload, "attempts": payload["attempts"][:-1]})


def _sample_json() -> dict[str, object]:
    return {
        "wrapper_step": 158,
        "records": [
            {
                "row_index": 100 + problem,
                "problem": f"Problem {problem}",
                "ground_truth": str(problem),
                "answer_style": "exact",
                "samples": [
                    {
                        "text": f"work {sample}\nAnswer: {problem}</s>",
                        "emitted_text": f"work\nAnswer: {problem}",
                        "trace": "tEE" if sample % 2 == 0 else "EtE",
                        "thinks": 1,
                        "emits": 2,
                        "correct": sample == 0,
                        "prediction": str(problem),
                        "terminated": sample != 3,
                        "forced_initial_think": sample % 2 == 0,
                        "optional_thought_count": 0 if sample % 2 == 0 else 1,
                        "think_run_lengths": [1],
                    }
                    for sample in range(4)
                ],
            }
            for problem in range(4)
        ],
    }


def test_sample_json_converter_preserves_exact_panel_and_computes_metrics(tmp_path):
    sampled = _sample_json()
    payload = sample_json_to_report_payload(sampled)

    assert payload["step"] == 158
    assert payload["attempts"][0]["answer_style"] == "exact"
    assert payload["metrics"]["accuracy"] == 0.25
    assert payload["metrics"]["forced_initial_accuracy"] == 0.5
    assert payload["metrics"]["unforced_initial_accuracy"] == 0.0
    assert payload["metrics"]["think_fraction"] == 8 / 40
    assert payload["attempts"][0]["action_trace"] == "TEE"
    assert payload["attempts"][-1]["dataset_index"] == 103

    source = tmp_path / "samples.json"
    output = tmp_path / "answers.html"
    source.write_text(json.dumps(sampled))
    assert write_sample_json_report(source, output) == output
    assert "What the model answered" in output.read_text()


def test_sample_json_converter_supports_full_rectangular_evaluations():
    sampled = _sample_json()
    sampled["records"] = sampled["records"][:2]
    for record in sampled["records"]:
        record["samples"] = record["samples"][:3]

    payload = sample_json_to_report_payload(sampled)
    report = render_benchmark_report(payload)

    assert payload["problem_count"] == 2
    assert payload["samples_per_problem"] == 3
    assert len(payload["attempts"]) == 6
    assert "2 dataset problems · 3 policy samples each" in report
    assert report.count('class="attempt ') == 6


def test_sample_json_converter_rejects_ragged_evaluations():
    sampled = _sample_json()
    sampled["records"][0]["samples"] = sampled["records"][0]["samples"][:3]

    with pytest.raises(ValueError, match="same number"):
        sample_json_to_report_payload(sampled)


def test_sample_json_converter_rejects_uniformly_truncated_evaluations():
    sampled = _sample_json()
    sampled["samples_per_problem"] = 4
    for record in sampled["records"]:
        record["samples"] = record["samples"][:3]

    with pytest.raises(ValueError, match=r"same number.*\(4\)"):
        sample_json_to_report_payload(sampled)


def test_periodic_writer_rejects_non_capture_rectangles(tmp_path):
    attempts = [
        _attempt(problem, sample)
        for problem in range(2)
        for sample in range(3)
    ]

    with pytest.raises(ValueError, match="exactly the first four"):
        write_benchmark_report(tmp_path, 80, _payload()["metrics"], attempts)
