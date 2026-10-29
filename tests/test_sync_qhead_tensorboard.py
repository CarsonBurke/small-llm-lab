from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from scripts import sync_qhead_tensorboard as bridge


class Writer:
    def __init__(self):
        self.points = []
        self.flushes = 0

    def add_scalar(self, tag, value, step):
        self.points.append((tag, step, value))

    def flush(self):
        self.flushes += 1


def actor(step, value=0.125):
    return {"type": "train", "step": step, "dashboard": {
        "value/online_epoch_explained_variance": value,
        "value/target_mean": 0.25,
        "value/residual_mean": 0.375,
        "value/action_value_online_explained_variance": 99,
        "value/action_value_residual_mean": 100,
    }}


def line(record):
    return (json.dumps(record) + "\n").encode()


def test_mapping_uses_v_mc_targets_and_only_missing_source_tags():
    mapped = bridge.scalar_mapping(actor(42))
    assert mapped == {"value/mc_explained_variance": 0.125,
                      "value/mc_target_mean": 0.25, "value/mc_residual_mean": 0.375}
    assert bridge.scalar_mapping({"type": "train", "step": 42, "dashboard": {}}) == {}
    record = {"type": "rollout", "step": 42, "source_metrics": {"science_mc": {
        "reward_mean": 1, "ended_fraction": 1, "think_tokens_mean": 80,
        "think_gate_zeroed_correct_fraction": 0.5,
    }}}
    assert bridge.scalar_mapping(record) == {
        "source/science_mc/think_tokens_mean": 80,
        "source/science_mc/think_gate_zeroed_correct_fraction": 0.5,
    }
    assert bridge.scalar_mapping({**actor(42), "type": "value_warmup"}) == {}
    assert bridge.scalar_mapping(actor(42, float("nan"))).keys() == {
        "value/mc_target_mean", "value/mc_residual_mean"}


def test_partial_lines_append_bounded_batches_and_durable_restart(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    first, second = line(actor(41)), line(actor(42))
    metrics.write_bytes(first + second[:20])
    writer = Writer()
    sync = bridge.MetricSync(tmp_path, writer, emitted={})
    assert sync.sync(max_lines=1) == 1
    assert sync.state["offset"] == len(first)
    assert sync.sync() == 0
    assert len(writer.points) == 3
    with metrics.open("ab") as stream:
        stream.write(second[20:] + line(actor(43)))
    assert sync.sync(max_lines=1) == 1
    assert [point[1] for point in writer.points] == [41] * 3 + [42] * 3
    restarted = bridge.MetricSync(tmp_path, writer, emitted={})
    assert restarted.sync() == 1
    assert [point[1] for point in writer.points] == [41] * 3 + [42] * 3 + [43] * 3
    assert restarted.sync() == 0


@pytest.mark.parametrize("change", ["truncate", "replace", "rewrite_regrow"])
def test_rewritten_metrics_fail_closed_without_modifying_state(tmp_path, change):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_bytes(line(actor(41)))
    writer = Writer()
    sync = bridge.MetricSync(tmp_path, writer, emitted={})
    sync.sync()
    saved_state = sync.state_path.read_bytes()
    if change == "truncate":
        metrics.write_bytes(b"")
    elif change == "replace":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(line(actor(41)))
        replacement.replace(metrics)
    else:
        metrics.write_bytes(line(actor(22)) + line(actor(23)))
    with pytest.raises(bridge.MetricsFileChanged):
        sync.sync()
    assert sync.state_path.read_bytes() == saved_state
    assert len(writer.points) == 3


def test_crash_after_event_flush_recovers_without_duplicate_points(tmp_path, monkeypatch):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_bytes(line(actor(75, 0.123456789)))
    directory = tmp_path / "tensorboard"
    writer = bridge.ScalarEventWriter(directory)
    real_atomic = bridge.atomic_json
    monkeypatch.setattr(bridge, "atomic_json", lambda *args: (_ for _ in ()).throw(OSError("crash")))
    try:
        sync = bridge.MetricSync(tmp_path, writer)
        with pytest.raises(OSError, match="crash"):
            sync.sync()
    finally:
        writer.close()
    assert not sync.state_path.exists()
    monkeypatch.setattr(bridge, "atomic_json", real_atomic)
    recovered_writer = Writer()
    recovered = bridge.MetricSync(tmp_path, recovered_writer)
    assert recovered.sync() == 1
    assert recovered_writer.points == []
    assert len(bridge.event_points(directory)) == 3


def test_provenance_rejects_q_critic_and_unverified_source(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"critic": {"critic_schema": "q_critic"}}))
    with pytest.raises(ValueError, match="scalar V"):
        bridge.source_provenance(tmp_path)
    core = b"verified target recurrence fixture"
    trainer = b"verified trainer target usage fixture"
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as snapshot:
        entry = tarfile.TarInfo("postraining/core.py")
        entry.size = len(core)
        snapshot.addfile(entry, io.BytesIO(core))
        entry = tarfile.TarInfo("postraining/train_latent_vapo.py")
        entry.size = len(trainer)
        snapshot.addfile(entry, io.BytesIO(trainer))
    manifest_path.write_text(json.dumps({
        "critic": {"critic_schema": "scalar_zero_init_linear_unclipped_mse/v1"},
        "source_provenance": {"source_archive": archive.name,
                              "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()},
    }))
    with pytest.raises(ValueError, match="verified original"):
        bridge.source_provenance(tmp_path)
    monkeypatch.setattr(bridge, "VERIFIED_CORE_SHA256", hashlib.sha256(core).hexdigest())
    with pytest.raises(ValueError, match="trainer target semantics"):
        bridge.source_provenance(tmp_path)
    monkeypatch.setattr(bridge, "VERIFIED_TRAINER_SHA256", hashlib.sha256(trainer).hexdigest())
    assert bridge.source_provenance(tmp_path)["value_aliases"] == bridge.VALUE_ALIASES
