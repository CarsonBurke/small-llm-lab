from __future__ import annotations

import sys

from scripts import ablation


def test_main_propagates_training_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["ablation.py", "--name", "failure_probe", "--steps", "1"],
    )
    monkeypatch.setattr(
        ablation,
        "run_config",
        lambda *args, **kwargs: {"returncode": 7},
    )
    assert ablation.main() == 7

