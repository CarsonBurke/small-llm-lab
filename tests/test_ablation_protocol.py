from __future__ import annotations

import json

import pytest

from scripts import ablation
from scripts.ablation import (
    REFERENCE_WARMDOWN_ITERS,
    build_run_env,
    compare_results,
    metric_integrity_errors,
    parse_extra_metrics,
    parse_log_line,
    parse_time_ms,
    read_metrics_jsonl,
    run_config,
)


def test_ablation_uses_reference_warmdown_instead_of_ambient_value(monkeypatch) -> None:
    monkeypatch.setenv("WARMDOWN_ITERS", "17")

    env = build_run_env({}, steps=2000, val_every=20, name="baseline_2k")

    assert env["WARMDOWN_ITERS"] == str(REFERENCE_WARMDOWN_ITERS)
    assert env["ITERATIONS"] == "2000"
    assert env["VAL_LOSS_EVERY"] == "20"


def test_ablation_allows_explicit_warmdown_override() -> None:
    env = build_run_env(
        {"WARMDOWN_ITERS": "0"},
        steps=2000,
        val_every=20,
        name="flat_lr_control",
    )

    assert env["WARMDOWN_ITERS"] == "0"


def test_short_checkpoint_run_keeps_flat_learning_rate() -> None:
    env = build_run_env({}, steps=200, val_every=20, name="short_control")

    assert env["WARMDOWN_ITERS"] == "0"


def test_warmdown_default_tracks_effective_iteration_override() -> None:
    short_env = build_run_env(
        {"ITERATIONS": "200"},
        steps=2000,
        val_every=20,
        name="short_override",
    )
    long_env = build_run_env(
        {"ITERATIONS": "2000"},
        steps=200,
        val_every=20,
        name="long_override",
    )

    assert short_env["WARMDOWN_ITERS"] == "0"
    assert long_env["WARMDOWN_ITERS"] == str(REFERENCE_WARMDOWN_ITERS)


def test_metric_parser_preserves_scientific_notation() -> None:
    entry = parse_log_line(
        "step:2000/2000 train_loss:1.25e+00 train_time:6.84796504e5ms "
        "lr:2.5e-07 signed:-3.125E+2 leading:.5 trailing:1."
    )

    assert entry is not None
    assert entry["train_loss"] == pytest.approx(1.25)
    assert entry["train_time_ms"] == pytest.approx(684_796.504)
    assert entry["lr"] == pytest.approx(2.5e-7)
    assert entry["signed"] == pytest.approx(-312.5)
    assert entry["leading"] == pytest.approx(0.5)
    assert entry["trailing"] == pytest.approx(1.0)


def test_numeric_helpers_do_not_truncate_exponents() -> None:
    assert parse_extra_metrics("small:7.525e-05 large:+1.2E3") == {
        "small": pytest.approx(7.525e-5),
        "large": pytest.approx(1_200.0),
    }
    assert parse_time_ms("train_time:6.0e-2m") == pytest.approx(3_600.0)
    assert parse_time_ms("train_time:1.5s") == pytest.approx(1_500.0)
    assert parse_time_ms("train_time:1e3ms") == pytest.approx(1_000.0)
    assert parse_time_ms("train_time:3msjunk") is None
    assert parse_time_ms("train_time:1e") is None


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (
            "step:20/20 train_loss:nan train_time:1ms",
            "non-finite training loss",
        ),
        (
            "step:20/20 val_loss:1.0 val_bpb:inf train_time:1ms",
            "non-finite validation loss or BPB",
        ),
        (
            "step:20/20 train_loss:1.0 train_time:1ms grad_norm:inf",
            "non-finite metric grad_norm:inf",
        ),
        (
            "step:20/20 train_loss:1e train_time:1ms",
            "malformed training or validation metric line",
        ),
        (
            "step:20/20 train_loss:1.0 train_time:3msjunk",
            "missing or malformed train_time",
        ),
        (
            "step:20/20 val_loss:1.0 val_bpb:2.0 train_time:3msjunk",
            "malformed train_time",
        ),
    ],
)
def test_nonfinite_or_malformed_metrics_are_structured_errors(
    line: str, message: str
) -> None:
    entry = parse_log_line(line)
    assert entry is not None
    assert entry["type"] == "metric_error"
    assert entry["step"] == 20
    assert entry["metric_integrity_error"] == message


def test_metric_integrity_requires_one_finite_final_validation() -> None:
    finite_initial = {
        "type": "val",
        "step": 0,
        "val_loss": 2.0,
        "val_bpb": 3.0,
        "train_time_ms": 0.0,
    }
    nonfinite_final = {
        "type": "metric_error",
        "step": 20,
        "metric_integrity_error": "non-finite validation loss or BPB",
    }
    errors = metric_integrity_errors([finite_initial, nonfinite_final], 20)
    assert errors == [
        "non-finite validation loss or BPB",
        "missing finite validation at requested step 20",
    ]

    finite_final = {
        "type": "val",
        "step": 20,
        "val_loss": 1.0,
        "val_bpb": 2.0,
        "train_time_ms": 1.0,
    }
    assert metric_integrity_errors([finite_initial, finite_final], 20) == []


def test_validation_without_time_uses_metrics_writer_fallback() -> None:
    entry = parse_log_line("20/20 val_loss: 1.25 val_bpb: 1.75")
    assert entry == {
        "step": 20,
        "val_loss": 1.25,
        "val_bpb": 1.75,
        "train_time_ms": 0.0,
        "type": "val",
    }


def test_metrics_reader_rejects_malformed_middle_record(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        '{"type":"val","step":0}\n'
        '{broken}\n'
        '{"type":"val","step":20}\n'
    )
    with pytest.raises(ValueError, match=r"metrics.jsonl:2"):
        read_metrics_jsonl(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_metrics_reader_rejects_nonstandard_nonfinite_json(
    tmp_path, constant: str
) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        '{"type":"val","step":20,"val_loss":'
        + constant
        + ',"val_bpb":2,"train_time_ms":1}\n'
    )
    with pytest.raises(ValueError, match=r"metrics.jsonl:1"):
        read_metrics_jsonl(path)


def test_metrics_reader_rejects_finite_syntax_that_overflows(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        '{"type":"val","step":20,"val_loss":1,"val_bpb":2,'
        '"train_time_ms":1,"optional":1e309}\n'
    )
    with pytest.raises(ValueError, match=r"metrics.jsonl:1"):
        read_metrics_jsonl(path)


def test_metric_integrity_rechecks_loaded_final_scalars() -> None:
    entry = {
        "type": "val",
        "step": 20,
        "val_loss": float("nan"),
        "val_bpb": 2.0,
        "train_time_ms": 1.0,
    }
    assert metric_integrity_errors([entry], 20) == [
        "final validation has non-finite or missing val_loss"
    ]


def test_run_config_fails_closed_on_nonfinite_final_metric(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "logs").mkdir()
    child = tmp_path / "emit_nonfinite.py"
    child.write_text(
        "print('step:0/20 val_loss: 2.0 val_bpb: 3.0')\n"
        "print('step:20/20 val_loss: nan val_bpb: nan')\n"
    )
    monkeypatch.setattr(ablation, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ablation, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(ablation, "TB_DIR", tmp_path / "tb")

    result = run_config("invalid_final", {}, 20, 20, child.name)

    assert result["training_returncode"] == 0
    assert result["returncode"] == ablation.METRIC_INTEGRITY_RETURN_CODE
    assert result["final_val_bpb"] is None
    assert result["final_proxy_val_bpb"] is None
    assert result["metric_integrity_errors"] == [
        "non-finite validation loss or BPB",
        "missing finite validation at requested step 20",
    ]
    persisted = json.loads(
        (tmp_path / "results" / "invalid_final" / "result.json").read_text()
    )
    assert persisted["returncode"] == ablation.METRIC_INTEGRITY_RETURN_CODE
    assert persisted["final_val_bpb"] is None


def test_compare_results_rejects_nonfinite_legacy_json(tmp_path) -> None:
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    (run_dir / "result.json").write_text(
        '{"name":"legacy","returncode":0,"final_val_bpb":NaN}\n'
    )
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        compare_results(tmp_path)


def test_compare_results_rejects_overflowed_legacy_json(tmp_path) -> None:
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    (run_dir / "result.json").write_text(
        '{"name":"legacy","returncode":0,"final_val_bpb":1e309}\n'
    )
    with pytest.raises(ValueError, match=r"non-finite JSON number"):
        compare_results(tmp_path)
