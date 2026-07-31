from __future__ import annotations

import subprocess
import sys

import pytest

from run_k3_context_curriculum import (
    curriculum_stages,
    relay_stage_output,
    validate_training_manifest,
)


def test_default_full_curriculum_boundaries() -> None:
    assert curriculum_stages(20_000) == (
        ("ctx2k", 15_000, 2048, 16),
        ("ctx4k", 18_750, 4096, 8),
        ("ctx8k", 20_000, 8192, 4),
    )


def test_curriculum_keeps_microbatch_token_count_constant() -> None:
    stages = curriculum_stages(8_000)
    assert {sequence_length * microbatch for _, _, sequence_length, microbatch in stages} == {
        32_768
    }


def test_curriculum_rejects_too_few_steps() -> None:
    with pytest.raises(ValueError, match="too short"):
        curriculum_stages(2)


def valid_manifest() -> dict:
    return {
        "train_batch_tokens": 524_288,
        "loader_aligned": True,
        "unique_stream_tokens": 20_000 * 524_288 + 1,
        "shards": [{"steps": 10_000}, {"steps": 10_000}],
    }


def test_manifest_accepts_exactly_one_pass() -> None:
    validate_training_manifest(valid_manifest(), 20_000)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"train_batch_tokens": 262_144}, "batch geometry"),
        ({"loader_aligned": False}, "loader-aligned"),
        ({"unique_stream_tokens": 524_288}, "too short"),
        ({"shards": [{"steps": 19_999}]}, "too few"),
    ],
)
def test_manifest_rejects_any_geometry_that_can_cycle(
    update: dict,
    message: str,
) -> None:
    manifest = valid_manifest()
    manifest.update(update)
    with pytest.raises(ValueError, match=message):
        validate_training_manifest(manifest, 20_000)


class RecordingMetrics:
    def __init__(self, fail: bool = False) -> None:
        self.entries: list[dict] = []
        self.fail = fail

    def write_entry(self, entry: dict) -> None:
        if self.fail:
            raise RuntimeError("metric failure")
        self.entries.append(entry)


def test_stage_relay_offsets_training_time() -> None:
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "print('step:10/20 train_loss:1.5 train_time:125ms')",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    metrics = RecordingMetrics()
    duration = relay_stage_output(process, metrics, 1_000.0)
    assert duration == 125.0
    assert metrics.entries[0]["train_time_ms"] == 1_125.0


def test_stage_relay_terminates_child_if_metric_writer_fails() -> None:
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            (
                "import time; "
                "print('step:10/20 train_loss:1.5 train_time:125ms', flush=True); "
                "time.sleep(60)"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    with pytest.raises(RuntimeError, match="metric failure"):
        relay_stage_output(process, RecordingMetrics(fail=True), 0.0)
    assert process.poll() is not None
