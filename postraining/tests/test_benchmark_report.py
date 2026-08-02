from __future__ import annotations

import json

import pytest

from postraining.benchmark_report import (
    BENCHMARK_ANSWER_SCHEMA,
    bench_json_to_report_payload,
    render_benchmark_report,
    sample_json_to_report_payload,
    write_sample_json_report,
    write_benchmark_report,
)
from postraining.core import emitted_display_segments


class _StubTokenizer:
    """Duck-typed tokenizer: ids below 900 decode as their codepoint."""

    think_open_id = 900
    think_close_id = 901
    answer_open_id = 902
    answer_close_id = 903

    _PIECES = {
        900: "<think>",
        901: "</think>",
        902: "<answer>",
        903: "</answer>",
        999: "<|endoftext|>",
    }

    def eos_id(self) -> int:
        return 999

    def bos_id(self) -> int:
        return 999

    def decode(self, ids) -> str:
        return "".join(chr(i) for i in ids)

    def id_to_piece(self, token_id: int) -> str:
        return self._PIECES[token_id]


def _segments() -> list[dict[str, object]]:
    return [
        {"kind": "prefix", "text": "Answer hint: "},
        {"kind": "special", "role": "think_open", "text": "<think>", "token_id": 900},
        {
            "kind": "text",
            "text": "<script>alert('no')</script>\nliteral <think> text",
        },
        {"kind": "special", "role": "think_close", "text": "</think>", "token_id": 901},
        {"kind": "special", "role": "answer_open", "text": "<answer>", "token_id": 902},
        {"kind": "text", "text": "42"},
        {"kind": "special", "role": "answer_close", "text": "</answer>", "token_id": 903},
        {"kind": "special", "role": "eos", "text": "<|endoftext|>", "token_id": 999},
    ]


def _attempt(problem: int, sample: int) -> dict[str, object]:
    return {
        "problem_index": problem,
        "dataset_index": f"dataset-{problem}",
        "sample_index": sample,
        "prompt": "Find <x> & explain",
        "ground_truth": "<42>",
        "answer_style": "exact",
        "emitted_text": "<script>alert('no')</script>\nAnswer: 42",
        "emitted_segments": _segments(),
        "parsed_answer": "42",
        "correct": sample == 0,
        "terminated": sample != 3,
        "termination_token_id": 2 if sample != 3 else None,
        "emitted_token_count": 8,
    }


def _payload() -> dict[str, object]:
    return {
        "schema": BENCHMARK_ANSWER_SCHEMA,
        "step": 80,
        "reward_schema": "test_reward/v1",
        "metrics": {
            "accuracy": 0.25,
            "policy_accuracy": 0.25,
            "prompt_any_correct_fraction": 0.5,
            "prompt_mixed_reward_fraction": 0.125,
            "within_group_reward_std": 0.03125,
            "dataset_modal_answer": "5",
            "dataset_modal_answer_accuracy": 0.03594,
            "samples": 1152,
            "sampling_schema": "global_rng_compacted_tail/v1",
            "finished_compaction": "compiled_tail_b16",
        },
        "attempts": [
            _attempt(problem, sample)
            for problem in reversed(range(4))
            for sample in reversed(range(4))
        ],
    }


def test_emitted_display_segments_split_on_ids_not_text():
    tokenizer = _StubTokenizer()

    typed_fence = [ord(c) for c in "a<think>b"]
    assert emitted_display_segments(typed_fence, tokenizer) == [
        {"kind": "text", "text": "a<think>b"}
    ]

    stream = [900, ord("4"), ord("2"), 901, 999]
    segments = emitted_display_segments(stream, tokenizer)
    assert [segment["kind"] for segment in segments] == [
        "special",
        "text",
        "special",
        "special",
    ]
    assert segments[0] == {
        "kind": "special",
        "role": "think_open",
        "text": "<think>",
        "token_id": 900,
        "source": "text",
    }
    assert segments[1]["text"] == "42"
    # BOS and EOS share GPT-2's <|endoftext|>; terminal role wins.
    assert segments[-1]["role"] == "eos"

    assert emitted_display_segments([ord("A")], tokenizer, kind="prefix") == [
        {"kind": "prefix", "text": "A"}
    ]
    # A special token inside a teacher-forced prefix keeps its provenance.
    prefix_special = emitted_display_segments([999], tokenizer, kind="prefix")
    assert prefix_special[0]["source"] == "prefix"


def test_render_benchmark_report_is_self_contained_and_escapes_model_text():
    report = render_benchmark_report(_payload())

    assert "What the model answered" in report
    assert "All-rollout accuracy</span><strong>25.00%" in report
    assert "Policy accuracy</span><strong>25.00%" in report
    assert "Prompts solved at least once</span><strong>50.00%" in report
    assert "Mixed-reward prompt groups</span><strong>12.50%" in report
    assert "Mean within-group reward std</span><strong>0.0312" in report
    assert "Best constant-answer baseline</span><strong>3.59%" in report
    assert "full-dataset modal answer: 5" in report
    assert "test_reward/v1" in report
    assert "global_rng_compacted_tail/v1" in report
    assert "compiled_tail_b16" in report
    assert "exact" in report
    assert report.count('class="attempt ') == 16
    assert "&lt;script&gt;alert" in report
    assert "<script>" not in report
    assert "https://" not in report
    assert "src=" not in report


def test_render_benchmark_report_keeps_special_tokens_visible():
    report = render_benchmark_report(_payload())

    # One chip per special segment per attempt plus the legend's exemplar —
    # the literal "<think>" typed inside a text segment must NOT mint one.
    for role in ("think_open", "think_close", "answer_open", "answer_close", "eos"):
        assert report.count(f'class="tok tok-{role}"') == 17
    assert "literal &lt;think&gt; text" in report
    # Fence spans tint the text between them; the prefix renders muted.
    assert 'class="txt in-think"' in report
    assert 'class="txt in-answer"' in report
    assert 'class="txt prefix"' in report
    # Emitted line breaks stay visible.
    assert '<span class="nl">' in report
    # The legend only accompanies segment-aware attempts.
    assert "graded answer span" in report
    assert '1/4 correct' in report


def test_render_benchmark_report_flags_broken_fences_and_suppresses_tints():
    payload = _payload()
    duplicated = {
        "kind": "special",
        "role": "answer_open",
        "text": "<answer>",
        "token_id": 902,
    }
    for attempt in payload["attempts"]:
        attempt["emitted_segments"] = attempt["emitted_segments"] + [duplicated]

    report = render_benchmark_report(payload)

    # Two answer opens: the grader's single_fence_span would reject this,
    # so the report must not dress it up as valid structure. The only
    # remaining tinted spans are the legend's own samples.
    assert report.count("broken fences") == 16
    assert report.count('class="txt in-think"') == 1
    assert report.count('class="txt in-answer"') == 1
    # Chips stay visible — only the validity implication is withdrawn.
    assert report.count('class="tok tok-answer_open"') == 33


def test_render_benchmark_report_degrades_on_malformed_segments():
    payload = _payload()
    for attempt in payload["attempts"]:
        attempt["emitted_segments"] = [
            "not a mapping",
            {"kind": "special", "role": 'x" incorrect'},
            {"kind": "text"},
            {"kind": "special", "role": "eos", "text": "<|endoftext|>"},
        ]

    report = render_benchmark_report(payload)

    # Unknown/hostile roles collapse to the generic chip class instead of
    # smuggling extra class tokens; missing text renders empty; non-mapping
    # entries are skipped; nothing raises.
    assert report.count('class="tok tok-special"') == 16
    assert 'incorrect"><' not in report.replace("badge incorrect", "")
    assert report.count('class="tok tok-eos"') == 17
    assert "not a mapping" not in report


def test_render_benchmark_report_marks_prefix_sourced_chips():
    payload = _payload()
    for attempt in payload["attempts"]:
        attempt["emitted_segments"][0] = {
            "kind": "special",
            "role": "eos",
            "text": "<|endoftext|>",
            "token_id": 999,
            "source": "prefix",
        }

    report = render_benchmark_report(payload)

    assert report.count('class="tok tok-eos prefix"') == 16


def test_render_benchmark_report_falls_back_to_stripped_text():
    payload = _payload()
    for attempt in payload["attempts"]:
        del attempt["emitted_segments"]

    report = render_benchmark_report(payload)

    assert "Answer: 42" in report
    assert 'class="tok ' not in report
    assert "graded answer span" not in report


def test_bench_json_payload_rebuilds_segments_for_legacy_attempts():
    payload = _payload()
    sentinel = _segments()[:1]
    for index, attempt in enumerate(payload["attempts"]):
        if index == 0:
            attempt["emitted_segments"] = sentinel
        else:
            del attempt["emitted_segments"]
        attempt["emitted_token_ids"] = [900, ord("x"), 901, 999]
        # v3 emitted_text was decode(prefix + emitted) with specials
        # stripped; the rebuild must re-derive the un-persisted prefix.
        attempt["emitted_text"] = "Answer hint: x"

    upgraded = bench_json_to_report_payload(payload, _StubTokenizer())

    rebuilt = upgraded["attempts"][1]["emitted_segments"]
    assert [segment["kind"] for segment in rebuilt] == [
        "prefix",
        "special",
        "text",
        "special",
        "special",
    ]
    assert rebuilt[0] == {"kind": "prefix", "text": "Answer hint: "}
    assert rebuilt[1]["role"] == "think_open"
    # Attempts that already carry segments pass through untouched, and the
    # source payload is never mutated.
    assert upgraded["attempts"][0]["emitted_segments"] == sentinel
    assert "emitted_segments" not in payload["attempts"][1]


def test_bench_json_payload_refuses_mismatched_tokenizer_registration():
    payload = _payload()
    for attempt in payload["attempts"]:
        del attempt["emitted_segments"]
        attempt["emitted_token_ids"] = [900, ord("x"), 901, 999]
        # The run's own tokenizer produced text our reconstruction cannot
        # reproduce — e.g. a SentencePiece artifact re-read as GPT-2.
        attempt["emitted_text"] = "completely different"

    with pytest.raises(ValueError, match="special-token registration"):
        bench_json_to_report_payload(payload, _StubTokenizer())


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
    assert machine_readable["attempts"][0]["emitted_segments"] == _segments()
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
                        "emits": 2,
                        "correct": sample == 0,
                        "prediction": str(problem),
                        "terminated": sample != 3,
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
    assert payload["metrics"]["policy_accuracy"] == 0.25
    assert payload["metrics"]["policy_samples"] == 16
    assert payload["attempts"][0]["emitted_token_count"] == 2
    assert payload["attempts"][-1]["dataset_index"] == 103

    source = tmp_path / "samples.json"
    output = tmp_path / "answers.html"
    source.write_text(json.dumps(sampled))
    assert write_sample_json_report(source, output) == output
    rendered = output.read_text()
    assert "What the model answered" in rendered
    assert "graded answer span" not in rendered


def test_sample_json_converter_carries_segments_when_present():
    sampled = _sample_json()
    sampled["records"][0]["samples"][0]["segments"] = [
        {"kind": "special", "role": "eos", "text": "<|endoftext|>", "token_id": 999}
    ]

    payload = sample_json_to_report_payload(sampled)

    assert payload["attempts"][0]["emitted_segments"][0]["role"] == "eos"
    assert "emitted_segments" not in payload["attempts"][1]


def test_sample_json_converter_carries_the_pinned_emit_flag():
    sampled = _sample_json()
    sampled["metrics"] = {"pin_emit": True}

    payload = sample_json_to_report_payload(sampled)

    assert payload["metrics"]["pin_emit"] is True


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
    assert "2 dataset problems · 3 rollout samples each" in report
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
