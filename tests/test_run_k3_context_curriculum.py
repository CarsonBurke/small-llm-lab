from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import run_k3_context_curriculum as runner

from scripts.run_k3_context_curriculum import (
    curriculum_stages,
    latent_moe_environment,
    relay_stage_output,
    validate_training_manifest,
)


def test_latent_moe_environment_is_compute_matched_k3_default() -> None:
    env = latent_moe_environment(
        enabled=True,
        num_experts=32,
        top_k=2,
        latent_dim=128,
        expert_hidden=256,
        shared_hidden=64,
        num_shared_experts=2,
        layer_indices="0,1,2,3,4,5,6,7",
        quantile_balance_interval=1,
    )
    assert env == {
        "MOE_NUM_EXPERTS": "32",
        "MOE_TOP_K": "2",
        "MOE_LATENT_DIM": "128",
        "MOE_EXPERT_HIDDEN": "256",
        "MOE_SHARED_HIDDEN": "64",
        "MOE_NUM_SHARED_EXPERTS": "2",
        "MOE_LAYER_INDICES": "0,1,2,3,4,5,6,7",
        "MOE_QB_INTERVAL": "1",
    }


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


def test_vocab_size_comes_from_the_dataset_and_is_padded() -> None:
    """Token ids mean nothing without the vocabulary that produced them.

    A shard stream carries no record of its own, so the width of the embedding
    table has to come from the dataset manifest. Defaulting instead would make
    a ToaST+TST corpus index out of range at best and silently make the tail of
    its vocabulary unreachable at worst.
    """
    assert runner.padded_vocab_size({}) == 50_304
    assert (
        runner.padded_vocab_size({"tokenizer_provenance": {"vocab_size": 50257}})
        == 50_304
    )
    assert (
        runner.padded_vocab_size({"tokenizer_provenance": {"vocab_size": 16384}})
        == 16_384
    )
    assert (
        runner.padded_vocab_size({"tokenizer_provenance": {"vocab_size": 16385}})
        == 16_512
    )


def write_dataset(directory: Path, manifest: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "mix_manifest.json").write_text(json.dumps(manifest))
    return directory


def toast_provenance(**overrides) -> dict:
    provenance = {
        "kind": "toast_tst",
        "name": "toast_tst_50k",
        "vocab_size": 50_257,
        "eot_id": 0,
        "directory": "data/tokenizers/toast_tst_50k",
        "spec_sha256": "a" * 64,
        "ngrams_sha256": "b" * 64,
    }
    provenance.update(overrides)
    return provenance


def curriculum_argv(main: Path, cooldown: Path) -> list[str]:
    return [
        "run_k3_context_curriculum.py",
        "--run-id",
        "identity_guard",
        "--steps",
        "20000",
        "--data",
        str(main),
        "--cooldown-data",
        str(cooldown),
    ]


def test_the_cooldown_dataset_must_share_the_main_vocabulary_exactly(
    tmp_path, monkeypatch
) -> None:
    """Two datasets, one embedding table.

    GPT-2's 50,257 tokens and a trained tokenizer's 50,257 tokens pad to the
    same 50,304, so comparing padded widths would let the final stage feed the
    model ids that mean something entirely different -- which reads as a
    catastrophic distribution shift rather than as the configuration error it
    is.
    """
    main = write_dataset(tmp_path / "main", valid_manifest() | {"tokenizer": "gpt2"})
    cooldown = write_dataset(
        tmp_path / "cooldown",
        valid_manifest()
        | {
            "tokenizer": "toast_tst_50k",
            "tokenizer_provenance": toast_provenance(),
            "training_steps": 20_000,
        },
    )
    monkeypatch.setattr(sys, "argv", curriculum_argv(main, cooldown))
    monkeypatch.chdir(tmp_path)
    assert runner.padded_vocab_size(
        json.loads((cooldown / "mix_manifest.json").read_text())
    ) == runner.padded_vocab_size(json.loads((main / "mix_manifest.json").read_text()))
    with pytest.raises(ValueError, match="exact vocabulary"):
        runner.main()
    # The guard has to fire before anything is launched or written.
    assert not (tmp_path / "ablation_results").exists()


def test_a_matching_cooldown_dataset_passes_the_vocabulary_guard(
    tmp_path, monkeypatch
) -> None:
    """The guard must not reject a renamed or relocated copy of one tokenizer."""
    manifest = valid_manifest() | {
        "tokenizer": "toast_tst_50k",
        "tokenizer_provenance": toast_provenance(),
    }
    main = write_dataset(tmp_path / "main", manifest)
    cooldown = write_dataset(
        tmp_path / "cooldown",
        manifest
        | {
            "tokenizer_provenance": toast_provenance(
                name="copy", directory="/elsewhere/copy"
            ),
            "training_steps": 20_000,
        },
    )
    monkeypatch.setattr(sys, "argv", curriculum_argv(main, cooldown))
    monkeypatch.chdir(tmp_path)

    def refuse_to_launch(*args, **kwargs):
        raise RuntimeError("stage launched")

    # Past the guard, main() goes on to launch training, which this test has
    # no business doing; reaching the launch is the evidence that the guard
    # passed.
    monkeypatch.setattr(runner.subprocess, "Popen", refuse_to_launch)
    with pytest.raises(RuntimeError, match="stage launched"):
        runner.main()
    assert (tmp_path / "ablation_results" / "identity_guard").exists()
