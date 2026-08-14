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
    assert env["ABLATION_RUNNER_OWNS_METRICS"] == "1"


def test_ablation_allows_explicit_warmdown_override() -> None:
    env = build_run_env(
        {"WARMDOWN_ITERS": "0"},
        steps=2000,
        val_every=20,
        name="flat_lr_control",
    )

    assert env["WARMDOWN_ITERS"] == "0"


def test_ablation_metrics_ownership_cannot_be_overridden() -> None:
    with pytest.raises(ValueError, match="reserved by the runner"):
        build_run_env(
            {"ABLATION_RUNNER_OWNS_METRICS": "0"},
            steps=20,
            val_every=20,
            name="invalid_metrics_owner",
        )


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


def test_unavailable_compiler_step_average_is_not_a_metric_error() -> None:
    entry = parse_log_line(
        "step:20/20 train_loss:1.0 train_time:100ms step_avg:nanms lr:1e-3"
    )

    assert entry is not None
    assert entry["type"] == "train"
    assert entry["train_loss"] == pytest.approx(1.0)
    assert entry["train_time_ms"] == pytest.approx(100.0)
    assert entry["lr"] == pytest.approx(1e-3)
    assert "step_avg_ms" not in entry


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


def test_diffusion_validation_does_not_masquerade_as_bpb() -> None:
    entry = parse_log_line(
        "step:20/20 val_loss:1.25 "
        "val_diffusion_nelbo_bits_per_atom:1.75 train_time:3ms"
    )
    assert entry == {
        "step": 20,
        "val_loss": 1.25,
        "val_diffusion_nelbo_bits_per_atom": 1.75,
        "train_time_ms": 3.0,
        "type": "val",
    }
    assert "val_bpb" not in entry


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


def test_run_config_forwards_explicit_training_script_arguments(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "logs").mkdir()
    child = tmp_path / "emit_args.py"
    child.write_text(
        "import sys\n"
        "assert sys.argv[1:] == ['--canvas-length', '256', '--branches', '15']\n"
        "print('step:0/20 val_loss:2.0 val_bpb:3.0')\n"
        "print('step:20/20 val_loss:1.0 val_bpb:2.0')\n"
    )
    monkeypatch.setattr(ablation, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ablation, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(ablation, "TB_DIR", tmp_path / "tb")

    script_args = ["--canvas-length", "256", "--branches", "15"]
    result = run_config("script_args", {}, 20, 20, child.name, script_args)

    assert result["returncode"] == 0
    assert result["script_args"] == script_args


def test_run_config_records_predeclared_futility_stop(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "logs").mkdir()
    child = tmp_path / "emit_futile.py"
    child.write_text(
        "import time\n"
        "print('step:0/2000 val_loss:3.0 "
        "val_diffusion_nelbo_bits_per_atom:4.0', flush=True)\n"
        "print('step:800/2000 val_loss:2.0 "
        "val_diffusion_nelbo_bits_per_atom:3.1', flush=True)\n"
        "time.sleep(60)\n"
    )
    monkeypatch.setattr(ablation, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ablation, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(ablation, "TB_DIR", tmp_path / "tb")

    gate = (800, "val_diffusion_nelbo_bits_per_atom", 3.05)
    result = run_config(
        "futile", {}, 2_000, 20, child.name, futility_gates=(gate,)
    )

    assert result["returncode"] == 0
    assert result["training_returncode"] != 0
    assert result["completed_steps"] == 800
    assert result["final_diffusion_nelbo_bits_per_atom"] == 3.1
    assert result["metric_integrity_errors"] == []
    assert result["early_stop"] == {
        "step": 800,
        "metric": "val_diffusion_nelbo_bits_per_atom",
        "observed": 3.1,
        "reject_at_or_above": 3.05,
        "decision": "rejected_for_futility",
        "termination": {
            "signal": 15,
            "forced": False,
            "returncode": -15,
        },
    }


def test_futility_stop_kills_sigterm_resistant_process_group(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "logs").mkdir()
    child = tmp_path / "ignore_term.py"
    child.write_text(
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('step:20/2000 val_loss:2.0 val_bpb:3.1', flush=True)\n"
        "time.sleep(60)\n"
    )
    monkeypatch.setattr(ablation, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ablation, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(ablation, "TB_DIR", tmp_path / "tb")
    monkeypatch.setattr(ablation, "TERMINATE_GRACE_SECONDS", 0.1)

    result = run_config(
        "force_kill", {}, 2_000, 20, child.name,
        futility_gates=((20, "val_bpb", 3.0),),
    )

    assert result["returncode"] == 0
    assert result["training_returncode"] == -9
    assert result["early_stop"]["termination"] == {
        "signal": 9,
        "forced": True,
        "returncode": -9,
    }


def test_futility_stop_kills_resistant_descendant_after_leader_exits(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "logs").mkdir()
    child = tmp_path / "resistant_descendant.py"
    child.write_text(
        "import subprocess, sys, time\n"
        "descendant = subprocess.Popen([sys.executable, '-c', "
        "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print(\"ready\", flush=True); time.sleep(60)'], "
        "stdout=subprocess.PIPE, text=True)\n"
        "assert descendant.stdout.readline().strip() == 'ready'\n"
        "print('step:20/2000 val_loss:2.0 val_bpb:3.1', flush=True)\n"
        "time.sleep(60)\n"
    )
    monkeypatch.setattr(ablation, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ablation, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(ablation, "TB_DIR", tmp_path / "tb")
    monkeypatch.setattr(ablation, "TERMINATE_GRACE_SECONDS", 0.1)

    result = run_config(
        "kill_descendant", {}, 2_000, 20, child.name,
        futility_gates=((20, "val_bpb", 3.0),),
    )

    assert result["returncode"] == 0
    assert result["training_returncode"] == -15
    assert result["early_stop"]["termination"] == {
        "signal": 9,
        "forced": True,
        "returncode": -15,
    }


def test_generation_primary_recipe_never_ranks_its_ar_anchor(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "logs").mkdir()
    child = tmp_path / "emit_idlm_anchor.py"
    child.write_text(
        "print('step:0/20 val_loss:2.0 val_bpb:3.0 "
        "val_ar_anchor_bpb:3.0 val_diffusion_nelbo_bits_per_atom:3.0 "
        "generation_primary:1')\n"
        "print('step:20/20 val_loss:1.0 val_bpb:2.0 "
        "val_ar_anchor_bpb:2.0 val_diffusion_nelbo_bits_per_atom:2.0 "
        "generation_primary:1')\n"
    )
    monkeypatch.setattr(ablation, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ablation, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(ablation, "TB_DIR", tmp_path / "tb")

    result = run_config("idlm", {}, 20, 20, child.name)

    assert result["returncode"] == 0
    assert result["final_val_bpb"] is None
    assert result["final_proxy_val_bpb"] is None
    assert result["final_ar_anchor_bpb"] == 2.0
    assert result["final_diffusion_nelbo_bits_per_atom"] == 2.0
    assert (
        result["promotion_metric"]
        == "gsm8k_exact_match_generation_accuracy"
    )


@pytest.mark.parametrize(
    "script_name",
    ("train_byte_idlm.py", "train_byte_diffusion_gemma.py", "train_byte_duo.py"),
)
def test_known_generation_recipe_fails_closed_even_if_flag_is_missing(
    tmp_path, monkeypatch, script_name: str
) -> None:
    (tmp_path / "logs").mkdir()
    child = tmp_path / script_name
    child.write_text(
        "print('step:20/20 val_loss:1.0 val_bpb:2.0 "
        "val_ar_anchor_bpb:2.0')\n"
    )
    monkeypatch.setattr(ablation, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ablation, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(ablation, "TB_DIR", tmp_path / "tb")

    result = run_config("idlm_missing_flag", {}, 20, 20, child.name)

    assert result["returncode"] == 0
    assert result["final_val_bpb"] is None
    assert result["final_proxy_val_bpb"] is None
    assert result["final_ar_anchor_bpb"] == 2.0
    assert result["promotion_metric"] == "gsm8k_exact_match_generation_accuracy"


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
