"""Check the SFT-to-RL artifact boundary without constructing a model."""
import json
from pathlib import Path

import pytest
import torch

from scripts import run_code_sft_rl as runner


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def artifacts(tmp_path):
    data = tmp_path / "postraining/data"
    data.mkdir(parents=True)
    traces = data / "code_sft_traces.parquet"
    traces.write_bytes(b"verified code training corpus")
    source = data / "code_rl.parquet"
    source.write_bytes(b"separate code RL tasks")
    mixture_path = data / "original.mixture.json"
    write_json(mixture_path, {"sources": [{
        "name": "ultradata_code_l3",
        "path": str(source.relative_to(tmp_path)),
        "sha256": runner.file_sha256(source),
    }]})
    audit_path = data / "separation.json"
    write_json(audit_path, {
        "rl_mixture": str(mixture_path.relative_to(tmp_path)),
        "rl_mixture_sha256": runner.file_sha256(mixture_path),
    })
    write_json(traces.with_suffix(".manifest.json"), {
        "output_sha256": runner.file_sha256(traces),
        "rl_separation_audit": str(audit_path.relative_to(tmp_path)),
        "rl_separation_audit_sha256": runner.file_sha256(audit_path),
    })
    settings = {
        "epochs": 1, "allow_unverified": False, "think_tokens": True,
        "answer_fence": True, "skip_sampling_gate": True,
    }
    run = tmp_path / "postraining/runs/code_sft"
    write_json(run / "result.json", {"args": settings, "steps": 1234})
    metadata = {
        "schema": "sft_trace_train/v1", "args": settings, "steps": 1234,
        "traces_sha256": runner.file_sha256(traces),
        "answer_fence_document_fraction": 1.0,
    }
    # Metadata-only torch archive: no model, tensors, GPU, or inference.
    torch.save({"sft": metadata}, run / "sft_final_model.pt")
    return tmp_path, traces, source, run, metadata


def test_completed_sft_binds_rl_corpus_and_total_update_budget(artifacts):
    root, traces, source, run, _ = artifacts
    command = runner.prepare_rl(root, "code_sft", "code_rl")

    def option(name):
        return command[command.index(name) + 1]

    assert option("--checkpoint") == str((run / "sft_final_model.pt").resolve())
    assert int(option("--steps")) == 950
    assert int(option("--value-warmup-steps")) == 50
    assert int(option("--steps")) + int(option("--value-warmup-steps")) == 1000
    assert option("--python-reward-mode") == "test-fraction"
    assert option("--continuation-tokens") == "2048"
    assert option("--critic-init") == "scratch"
    assert option("--critic-q-head-hidden") == "1024"
    assert option("--q-critic-lambda") == "1.0"
    assert option("--snapshot-every") == "50"
    assert "--critic-action-head-hidden" not in command
    assert "--actor-init" not in command
    assert "--fresh-critic" not in command
    assert "--value-warmup-rewind" not in command
    from postraining.vapo.config import build_arg_parser, validate_args
    parser = build_arg_parser()
    args = parser.parse_args(command[4:])
    validate_args(parser, args)
    assert option("--output") == str(root / "postraining/runs/code_rl")
    manifest = json.loads(Path(option("--rl-mixture-manifest")).read_text())
    assert manifest["sft_corpus"] == str(traces.resolve())
    assert manifest["sft_corpus_sha256"] == runner.file_sha256(traces)
    assert manifest["sources"][0]["path"] == str(source.resolve())
    assert manifest["code_sft_lineage"]["checkpoint_sha256"] == runner.file_sha256(
        run / "sft_final_model.pt")
    assert manifest["code_sft_lineage"]["optimizer_updates"] == 1234
    assert manifest["code_sft_lineage"]["rl_critic_family"] == "standalone_token_q"
    assert manifest["code_sft_lineage"]["planned_combined_optimizer_updates"] == 2234
    assert manifest["code_sft_lineage"]["warmup_prompt_policy"] == "consume_first_3200_then_continue"


def test_checkpoint_from_another_training_corpus_is_rejected(artifacts):
    root, _, _, run, metadata = artifacts
    metadata["traces_sha256"] = "0" * 64
    torch.save({"sft": metadata}, run / "sft_final_model.pt")
    with pytest.raises(ValueError, match="does not match its corpus/result"):
        runner.prepare_rl(root, "code_sft", "code_rl")
    assert not (root / "postraining/data/code_rl.mixture.manifest.json").exists()


def test_changed_rl_tasks_invalidate_separation_audit(artifacts):
    root, _, source, _, _ = artifacts
    source.write_bytes(b"changed tasks after overlap checks")
    with pytest.raises(ValueError, match="RL source bytes changed"):
        runner.prepare_rl(root, "code_sft", "code_rl")
    assert not (root / "postraining/data/code_rl.mixture.manifest.json").exists()


def test_follow_on_requires_code_tasks(artifacts):
    root, traces, _, _, _ = artifacts
    audit_path = root / "postraining/data/separation.json"
    audit = json.loads(audit_path.read_text())
    mixture_path = root / audit["rl_mixture"]
    mixture = json.loads(mixture_path.read_text())
    mixture["sources"][0]["name"] = "math_only"
    write_json(mixture_path, mixture)
    audit["rl_mixture_sha256"] = runner.file_sha256(mixture_path)
    write_json(audit_path, audit)
    manifest = json.loads(traces.with_suffix(".manifest.json").read_text())
    manifest["rl_separation_audit_sha256"] = runner.file_sha256(audit_path)
    write_json(traces.with_suffix(".manifest.json"), manifest)
    with pytest.raises(ValueError, match="must include audited code"):
        runner.prepare_rl(root, "code_sft", "code_rl")
    assert not (root / "postraining/data/code_rl.mixture.manifest.json").exists()


def test_existing_output_does_not_create_an_orphan_manifest(artifacts):
    root, *_ = artifacts
    (root / "postraining/runs/code_rl").mkdir()
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        runner.prepare_rl(root, "code_sft", "code_rl")
    assert not (root / "postraining/data/code_rl.mixture.manifest.json").exists()
