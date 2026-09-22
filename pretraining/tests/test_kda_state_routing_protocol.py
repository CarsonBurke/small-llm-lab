"""Read-only/static run-contract tests; no model or GPU execution."""

import ast
from pathlib import Path
import sys

import pytest

from scripts import train_kda_state_routing as runner


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = (
    ROOT / "pretraining/nanogpt_mini/nanogpt_mini_gpt2vocab_kda_3to1_pm_train.py"
)
FORK = ROOT / "pretraining/state_routing/train.py"


def definition(path, name):
    return next(
        node
        for node in ast.parse(path.read_text()).body
        if getattr(node, "name", None) == name
    )


@pytest.mark.parametrize(
    "name",
    [
        "CausalSelfAttention",
        "Rotary",
        "MLP",
        "initialize_model",
        "Muon",
        "distributed_data_generator",
    ],
)
def test_reference_math_initialization_optimizer_and_data_are_preserved(name):
    assert ast.dump(definition(REFERENCE, name)) == ast.dump(definition(FORK, name))


@pytest.mark.parametrize(
    "namespace", ["ablation_results/trial", "tb_logs/trial", "logs/trial.txt"]
)
def test_refuses_every_preexisting_run_namespace(tmp_path, monkeypatch, namespace):
    target = tmp_path / namespace
    target.parent.mkdir(parents=True)
    target.mkdir() if target.suffix != ".txt" else target.touch()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["runner", "--name", "trial", "--steps", "1000"])
    with pytest.raises(SystemExit) as stopped:
        runner.main()
    assert stopped.value.code == 2


@pytest.mark.parametrize(
    "arguments",
    [
        ["--steps", "2000"],
        ["--steps", "0"],
        ["--steps", "1000", "--seconds", "601"],
    ],
)
def test_rejects_budget_expansion(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["runner", "--name", "trial", *arguments])
    with pytest.raises(SystemExit) as stopped:
        runner.main()
    assert stopped.value.code == 2


def test_no_private_state_checkpoint_export_remains():
    source = FORK.read_text()
    assert "_final_model.pt" not in source
    assert "optimizer_update_in_progress = True" in source
    assert "completed_updates = step + 1" in source
