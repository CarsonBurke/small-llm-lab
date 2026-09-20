"""CPU contract of scripts/compare_stationary_stream.py over fabricated run artifacts."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("compare_stationary_stream",
                                              REPO_ROOT / "scripts/compare_stationary_stream.py")
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)

SHARED = {"data_path": "d", "seed": 1, "learning_rate": 1e-4, "document_batch": 4096, "stream_steps": 8,
          "iterations": 2000}
STATIONARY_SOURCES = ("pretraining/stationary_stream/config.py", "pretraining/stationary_stream/model.py",
                      "pretraining/stationary_stream/objective.py", "pretraining/stationary_stream/training.py")


def _write(root: Path, run: str, architecture: str, config: dict, sources: dict, bpb: float, gain: float,
           step_ms: float, parameters: int = 100) -> None:
    directory = root / run
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(json.dumps({
        "status": "completed", "steps": 2000, "tokens_seen": 65536000, "training_seconds": 50.0,
        "validation": {"val_bpb": bpb, "val_bpb_reset_latent": bpb + gain, "val_memory_gain_bpb": gain}}))
    (directory / "run_config.json").write_text(json.dumps({
        "architecture": architecture, "config": {"run_id": run, **config}, "sources": sources,
        "tokenizer_sha256": "tok", "train_fingerprint": "train", "val_fingerprint": "val", "parameters": parameters}))
    rows = [{"type": "train", "step": step, "step_avg_ms": step_ms, "null_mass": 0.0, "read_age": 1.0, "read_rms": 1.0}
            for step in range(100, 2100, 100)]
    (directory / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "RESULTS", tmp_path)
    control_sources = {key: "shared" for key in compare.CONTROL_SOURCES}
    old = {**control_sources, **{key: "old" for key in STATIONARY_SOURCES}}
    new = {**control_sources, **{key: "new" for key in STATIONARY_SOURCES}}
    read = {"buffer_slots": 10, "read_heads": 4, "read_key_dim": 32, "read_entry": "latent_norm"}
    _write(tmp_path, "control", "streaming_ffn_future_bag_carry_v1",
           {**SHARED, "objective": "ce", "horizon": 32}, control_sources, 2.10, 1.10, 20.0)
    _write(tmp_path, "old_arm", "streaming_ffn_stationary_buffer_v1", {**SHARED, **read}, old, 2.11, 1.08, 22.0)
    _write(tmp_path, "steep", "streaming_ffn_stationary_buffer_v1",
           {**SHARED, **read, "read_recency_slope": 4.0}, new, 2.09, 1.12, 22.0)
    _write(tmp_path, "unit_slope", "streaming_ffn_stationary_buffer_v1",
           {**SHARED, **read, "read_recency_slope": 1.0}, new, 2.12, 1.05, 22.0)
    return tmp_path


def test_reference_predating_a_knob_compares_at_its_default_and_reports_source_drift(results):
    candidate, old_arm = compare.load_run("steep"), compare.load_run("old_arm")
    differences = compare.check_reference(candidate, old_arm)
    assert differences["read_recency_slope"] == [4.0, 1.0]
    assert set(differences["sources"]) == set(STATIONARY_SOURCES)
    assert set(differences) == {"run_id", "read_recency_slope", "sources"}
    # An arm that ran at the default explicitly is the same arm as one that predates the knob.
    assert "read_recency_slope" not in compare.check_reference(compare.load_run("unit_slope"), old_arm)


def test_reference_must_share_backbone_sources_and_training_fields(results):
    candidate = compare.load_run("steep")
    drifted = compare.load_run("old_arm")
    drifted["sources"]["train_gpt.py"] = "other"
    with pytest.raises(SystemExit, match="shared backbone or data sources"):
        compare.check_reference(candidate, drifted)
    retrained = compare.load_run("old_arm")
    retrained["config"]["learning_rate"] = 3e-4
    with pytest.raises(SystemExit, match="configuration differs"):
        compare.check_reference(candidate, retrained)


def test_control_is_matched_on_shared_fields_and_reports_the_rest(results):
    differences = compare.check_control(compare.load_run("steep"), compare.load_run("control"))
    assert differences["read_recency_slope"] == [4.0, None] and differences["horizon"] == [None, 32]
    assert "learning_rate" not in differences


def test_main_writes_the_gate_and_arms(results, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["compare", "--candidate", "steep", "--control", "control",
                                     "--reference", "old_arm", "--reference", "unit_slope"])
    compare.main()
    summary = json.loads((results / "steep/comparison.json").read_text())
    assert summary["improvement_bpb"] == pytest.approx(0.01) and summary["meets_proxy_promotion_gate"]
    assert summary["candidate_over_control_update_time"] == pytest.approx(1.1)
    assert set(summary["arms"]) == {"steep", "control", "old_arm", "unit_slope"}
    assert summary["config_differences"]["old_arm"]["sources"]
    assert json.loads(capsys.readouterr().out)["candidate"] == "steep"
