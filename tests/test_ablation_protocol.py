from __future__ import annotations

from ablation import REFERENCE_WARMDOWN_ITERS, build_run_env


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
