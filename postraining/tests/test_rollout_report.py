from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from postraining.latent_rollout import PAD_SLOT, TOKEN_SLOT
from postraining.rollout_report import (
    REPORT_NAME,
    ROLLOUT_SAMPLE_SCHEMA,
    RolloutSampleRecorder,
    RowVerdict,
    capture_due,
    load_json_captures,
    main,
    parse_tensorboard_sample,
    purge_rollout_samples_from,
    render_rollout_report,
    write_rollout_samples,
)
from postraining.tests.test_benchmark_report import _StubTokenizer

EOS = 999


def _batch(rows: list[list[int]], rewards: list[float], prompt_length: int = 2):
    """Duck-typed scored group: prompt tokens, then right-padded emissions."""
    width = prompt_length + max(len(row) for row in rows)
    kind = torch.full((len(rows), width), PAD_SLOT, dtype=torch.long)
    token_ids = torch.zeros((len(rows), width), dtype=torch.long)
    kind[:, :prompt_length] = TOKEN_SLOT
    for index, row in enumerate(rows):
        kind[index, prompt_length : prompt_length + len(row)] = TOKEN_SLOT
        token_ids[index, prompt_length : prompt_length + len(row)] = torch.tensor(row)
    return SimpleNamespace(
        kind=kind,
        token_ids=token_ids,
        prompt_length=prompt_length,
        reward_scalar=torch.tensor(rewards),
    )


def _fenced(answer: str) -> list[int]:
    return [900, *map(ord, "work"), 901, 902, *map(ord, answer), 903, EOS]


def _recorder(sources=("dapo", "deepmind"), prefix=()) -> RolloutSampleRecorder:
    return RolloutSampleRecorder(_StubTokenizer(), (EOS,), sources, prefix)


def _verdicts(batch, answers=None) -> list[RowVerdict]:
    """Scorer-shaped verdicts; the recorder must pass them through as is."""
    rows = batch.reward_scalar.numel()
    answers = answers or [f"graded-{row}" for row in range(rows)]
    return [
        RowVerdict(format_ok=row % 2 == 0, parsed_answer=answers[row])
        for row in range(rows)
    ]


def _offer(recorder, batch, prompt, truth, source, answers=None) -> None:
    recorder.offer(batch, _verdicts(batch, answers), prompt, truth, source)


def test_capture_due_covers_every_pool_spanning_a_multiple() -> None:
    assert capture_due(0, 1, 25)
    assert not capture_due(1, 1, 25)
    assert capture_due(25, 1, 25)
    # A four-update pool starting at 23 covers step 25.
    assert capture_due(23, 4, 25)
    assert not capture_due(26, 4, 25)
    assert not capture_due(0, 1, 0)


def test_recorder_keeps_one_correct_and_incorrect_per_source() -> None:
    recorder = _recorder()
    _offer(recorder, _batch([_fenced("1")], [0.0]), "p", "t", "dapo")  # unarmed
    recorder.begin(50, metrics_step=54)
    _offer(
        recorder,
        _batch([_fenced("1"), _fenced("2"), [*map(ord, "no stop")]], [0.05, 1.0, 0.0]),
        "first prompt",
        "2",
        "dapo",
        answers=["1", "2", None],
    )
    # deepmind yields only an incorrect trajectory in this pool.
    _offer(recorder, _batch([[*map(ord, "abc")]], [0.0]), "dm prompt", "7", "deepmind")
    payload = recorder.finish()
    assert not recorder.armed
    assert payload["schema"] == ROLLOUT_SAMPLE_SCHEMA
    assert (payload["step"], payload["metrics_step"]) == (50, 54)
    assert payload["groups_offered"] == {"dapo": 1, "deepmind": 1}
    picked = {(s["source"], s["label"]): s for s in payload["samples"]}
    assert set(picked) == {
        ("dapo", "correct"),
        ("dapo", "incorrect"),
        ("deepmind", "incorrect"),
    }
    correct = picked[("dapo", "correct")]
    assert correct["prompt"] == "first prompt"
    assert correct["reward"] == 1.0
    assert (correct["group_correct"], correct["group_size"]) == (1, 3)
    assert correct["terminated"]
    # The scorer's verdict for row 1, verbatim.
    assert correct["structural_format_ok"] is False
    assert correct["parsed_answer"] == "2"
    roles = [s.get("role") for s in correct["emitted_segments"] if s["kind"] == "special"]
    assert roles == ["think_open", "think_close", "answer_open", "answer_close", "eos"]
    # Proximity credit below 1 is incorrect.
    incorrect = picked[("dapo", "incorrect")]
    assert incorrect["reward"] in (pytest.approx(0.05), 0.0)
    unterminated = picked[("deepmind", "incorrect")]
    assert not unterminated["terminated"]


def _pool_picks(order: list[int]) -> dict[str, str]:
    recorder = _recorder()
    recorder.begin(7, metrics_step=8)
    for index in order:
        _offer(
            recorder,
            _batch([_fenced(str(index)), _fenced("x")], [1.0, 0.0]),
            f"prompt {index}",
            str(index),
            "dapo",
        )
    return {
        sample["label"]: sample["prompt"] for sample in recorder.finish()["samples"]
    }


def test_recorder_picks_are_uniform_hashes_not_scoring_order() -> None:
    forward = _pool_picks(list(range(40)))
    # The collector scores shortest prompts first; the pick must not depend
    # on the order groups arrive in.
    assert _pool_picks(list(reversed(range(40)))) == forward
    # First-match would always show "prompt 0"; 40 prompts make a hash
    # landing on it for both labels a 1-in-1600 coincidence.
    assert forward != {"correct": "prompt 0", "incorrect": "prompt 0"}


def test_recorder_shows_the_teacher_forced_prefix() -> None:
    prefix = [*map(ord, "Answer:")]
    recorder = _recorder(prefix=prefix)
    recorder.begin(0, metrics_step=1)
    emitted = [*map(ord, " 4"), EOS]
    _offer(recorder, _batch([emitted], [1.0]), "p", "4", "dapo")
    (sample,) = recorder.finish()["samples"]
    # The verifier's decode: prefix, then the emission through its stop.
    assert sample["emitted_text"] == _StubTokenizer().decode(prefix + emitted)
    assert sample["emitted_segments"][0] == {"kind": "prefix", "text": "Answer:"}
    # The prefix was teacher-forced, not emitted.
    assert sample["emitted_token_count"] == 3


def test_recorder_refuses_bad_offers_and_double_arming() -> None:
    recorder = _recorder()
    with pytest.raises(ValueError, match="precedes"):
        recorder.begin(5, metrics_step=4)
    recorder.begin(0, metrics_step=1)
    with pytest.raises(RuntimeError, match="already armed"):
        recorder.begin(1, metrics_step=2)
    batch = _batch([_fenced("1")], [1.0])
    with pytest.raises(ValueError, match="unregistered"):
        _offer(recorder, batch, "p", "t", "gsm8k")
    with pytest.raises(ValueError, match="1-row group"):
        recorder.offer(batch, [], "p", "t", "dapo")
    recorder.finish()
    with pytest.raises(RuntimeError, match="never armed"):
        recorder.finish()


def _capture(step: int) -> dict:
    recorder = _recorder()
    recorder.begin(step, metrics_step=step + 1)
    _offer(
        recorder,
        _batch([_fenced("4"), _fenced("5")], [1.0, 0.0]),
        "What is <b>2+2</b>?",
        "4",
        "dapo",
    )
    return recorder.finish()


def test_write_purge_and_render_round_trip(tmp_path) -> None:
    for step in (0, 25, 50):
        report = write_rollout_samples(tmp_path, _capture(step))
    assert report == tmp_path / REPORT_NAME
    document = report.read_text()
    assert "Step 50" in document and "Step 0" in document
    assert "pool metrics logged at step 51" in document
    # Prompts are data: escaped, never markup.
    assert "<b>2+2</b>" not in document
    assert "&lt;b&gt;2+2&lt;/b&gt;" in document
    # A registered source absent from the pool says so instead of vanishing.
    assert "deepmind" in document and "0 groups in this pool" in document

    # A resume at step 25 re-collects that pool: 25 and 50 go.
    assert purge_rollout_samples_from(tmp_path, 25) == 2
    assert [c["step"] for c in load_json_captures(tmp_path)] == [0]
    assert "Step 25" not in (tmp_path / REPORT_NAME).read_text()
    assert purge_rollout_samples_from(tmp_path, 0) == 1
    assert not (tmp_path / REPORT_NAME).exists()


def test_json_loader_refuses_foreign_schemas(tmp_path) -> None:
    write_rollout_samples(tmp_path, _capture(0))
    path = tmp_path / "rollout_samples" / "step_000000.json"
    payload = json.loads(path.read_text())
    payload["schema"] = "something_else/v1"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema"):
        load_json_captures(tmp_path)


def test_render_requires_known_comparison_steps(tmp_path) -> None:
    with pytest.raises(ValueError, match="comparison steps"):
        render_rollout_report(
            [_capture(0)], run=tmp_path, last=4, compare_steps={7}
        )
    document = render_rollout_report(
        [_capture(0), _capture(25), _capture(50)],
        run=tmp_path,
        last=1,
        compare_steps={0},
    )
    assert 'id="step-50"' in document and 'details id="step-0"' in document
    assert 'id="step-25"' not in document


def test_cli_renders_a_kda_run(tmp_path, capsys) -> None:
    write_rollout_samples(tmp_path, _capture(0))
    (tmp_path / REPORT_NAME).unlink()
    output = tmp_path / "custom.html"
    assert main(["--run", str(tmp_path), "--output", str(output)]) == 0
    assert "Training rollouts" in output.read_text()
    assert "latest captured step 0" in capsys.readouterr().out


_CURRENT_SAMPLE = """Reward: 1

Response tokens: 42

Response limit: 100

Prompt tokens: 7

Repetition: {"repetition_3gram_fraction": 0.0}

Prompt:
What is 2 + 2?

Ground truth:
4

Model response:
The answer is 4."""

_LEGACY_SAMPLE = """Reward: -1

Response tokens: 12

Prompt:
What is 2 + 2?

Ground truth:
4

Model response:
5"""

_TRUNCATED_SAMPLE = """Reward: -1

Response tokens: 4000

Response limit: 4644

Prompt tokens: 5356

Repetition: {}

Prompt:
long prompt prefix

[... middle truncated ...]

response suffix"""


def test_tensorboard_sample_accepts_current_metadata_headers() -> None:
    sample = parse_tensorboard_sample(_CURRENT_SAMPLE, step=8, label="correct")
    assert sample["emitted_token_count"] == 42
    assert sample["prompt"] == "What is 2 + 2?"
    assert sample["ground_truth"] == "4"
    assert sample["emitted_text"] == "The answer is 4."
    assert sample["reward"] == 1.0


def test_tensorboard_sample_keeps_legacy_header_format() -> None:
    sample = parse_tensorboard_sample(_LEGACY_SAMPLE, step=8, label="incorrect")
    assert sample["emitted_token_count"] == 12
    assert sample["emitted_text"] == "5"


def test_tensorboard_sample_accepts_middle_truncation() -> None:
    sample = parse_tensorboard_sample(_TRUNCATED_SAMPLE, step=40, label="incorrect")
    assert sample["middle_truncated"]
    assert sample["prompt"].endswith("[... middle truncated ...]")
    assert sample["ground_truth"] == "[omitted from truncated TensorBoard sample]"
    assert sample["emitted_text"] == "response suffix"


def test_tensorboard_sample_rejects_bad_headers_and_labels() -> None:
    with pytest.raises(ValueError, match="unrecognized saved sample headers"):
        parse_tensorboard_sample(
            "Reward: 1\n\nResponse tokens: 3\n\nPrompt:\nmissing",
            step=0,
            label="correct",
        )
    with pytest.raises(ValueError, match="disagrees"):
        parse_tensorboard_sample(_LEGACY_SAMPLE, step=0, label="correct")


def test_cli_names_a_minicpm_run_by_its_event_directory_parent(tmp_path) -> None:
    (tmp_path / "tensorboard").mkdir()
    with pytest.raises(SystemExit):
        main(["--run", str(tmp_path / "tensorboard")])
    # No samples exist, so the error names the run the report would go to.
    with pytest.raises(ValueError, match=f"under {tmp_path}$"):
        from postraining.rollout_report import load_captures

        load_captures(tmp_path)
