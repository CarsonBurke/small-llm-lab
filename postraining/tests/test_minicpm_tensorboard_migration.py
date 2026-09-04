from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto.event_pb2 import Event, SessionLog
from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.compat.proto.types_pb2 import DT_FLOAT, DT_STRING
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from postraining.minicpm_tensorboard_schema import (
    EXPECTED_LEGACY_SCALAR_TAGS,
    EXPECTED_LEGACY_TEXT_TAGS,
    LEGACY_SCALAR_TAG_MAP,
    LEGACY_TAG_MAP,
    LEGACY_TEXT_TAG_MAP,
    MAX_SCALAR_TAGS_PER_CATEGORY,
    organized_tag,
    validate_scalar_category_cap,
)
from scripts import migrate_minicpm_tensorboard as migration


def _text_value(tag: str, text: str) -> Summary.Value:
    return Summary.Value(
        tag=tag,
        metadata=SummaryMetadata(
            plugin_data=SummaryMetadata.PluginData(plugin_name="text")
        ),
        tensor=TensorProto(
            dtype=DT_STRING,
            tensor_shape=TensorShapeProto(
                dim=[TensorShapeProto.Dim(size=1)]
            ),
            string_val=[text.encode("utf-8")],
        ),
    )


def _tensor_scalar_value(tag: str, value: float) -> Summary.Value:
    return Summary.Value(
        tag=tag,
        metadata=SummaryMetadata(
            display_name="raw tensor scalar",
            summary_description="metadata must survive byte-for-byte",
        ),
        tensor=TensorProto(
            dtype=DT_FLOAT,
            tensor_shape=TensorShapeProto(),
            float_val=[value],
        ),
    )


def _write_event_file(directory: Path, events: list[Event]) -> None:
    writer = EventFileWriter(str(directory), max_queue_size=1)
    try:
        for event in events:
            writer.add_event(event)
    finally:
        writer.close()


def _source_directory(tmp_path: Path) -> Path:
    directory = tmp_path / "tensorboard"
    _write_event_file(
        directory,
        [
            Event(
                wall_time=100.0,
                step=0,
                summary=Summary(
                    value=[
                        Summary.Value(tag="config/steps", simple_value=250.0),
                        _text_value(
                            "config/arguments/text_summary",
                            '{"steps": 250, "restart": false}',
                        ),
                    ]
                ),
            ),
            Event(
                wall_time=101.0,
                step=7,
                summary=Summary(
                    value=[
                        Summary.Value(tag="rollout/accuracy", simple_value=0.25)
                    ]
                ),
            ),
            Event(
                wall_time=102.0,
                step=7,
                summary=Summary(
                    value=[
                        Summary.Value(tag="rollout/accuracy", simple_value=0.75)
                    ]
                ),
            ),
            Event(
                wall_time=103.0,
                step=7,
                session_log=SessionLog(status=SessionLog.START),
            ),
        ],
    )
    _write_event_file(
        directory,
        [
            Event(
                wall_time=200.0,
                step=0,
                summary=Summary(
                    value=[
                        Summary.Value(tag="config/steps", simple_value=250.0)
                    ]
                ),
            ),
            Event(
                wall_time=201.0,
                step=8,
                summary=Summary(
                    value=[
                        _tensor_scalar_value("train/loss", 1.25),
                        _text_value(
                            "rollout_samples/correct/text_summary",
                            "correct sample",
                        ),
                    ]
                ),
            ),
            Event(
                wall_time=202.0,
                step=8,
                summary=Summary(
                    value=[
                        _text_value(
                            "rollout_samples/incorrect/text_summary",
                            "incorrect sample",
                        )
                    ]
                ),
            ),
        ],
    )
    return directory


def _event_paths(directory: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            directory.glob("events.out.tfevents.*"),
            key=migration._event_file_sort_key,
        )
    )


def _events(directory: Path) -> tuple[Event, ...]:
    events = []
    for path in _event_paths(directory):
        for raw in RawEventFileLoader(str(path)).Load():
            event = Event()
            event.ParseFromString(raw)
            events.append(event)
    return tuple(events)


def _organized_source_events(directory: Path) -> tuple[bytes, ...]:
    organized = []
    for event in _events(directory):
        cloned = Event()
        cloned.CopyFrom(event)
        for value in cloned.summary.value:
            value.tag = organized_tag(value.tag)
        organized.append(cloned.SerializeToString(deterministic=True))
    return tuple(organized)



def _all_accumulator_tags(directory: Path) -> set[str]:
    accumulator = EventAccumulator(str(directory))
    accumulator.Reload()
    return {
        tag
        for values in accumulator.Tags().values()
        if isinstance(values, list)
        for tag in values
    }

def _expected_prefix_mapping(
    source_prefix: str,
    destination_prefix: str,
    suffixes: str,
) -> dict[str, str]:
    return {
        f"{source_prefix}/{suffix}": f"{destination_prefix}/{suffix}"
        for suffix in suffixes.split()
    }


def _expected_legacy_scalar_map() -> dict[str, str]:
    groups = (
        (
            "config",
            "configuration_model",
            """
            trainable_actor_parameters critic_parameters critic_width lora_rank
            lora_alpha frozen_base_parameters total_policy_parameters
            rollout_replica_parameters rollout_fused_projection_groups
            estimated_static_kv_cache_bytes
            """,
        ),
        (
            "config",
            "configuration_nextlat",
            """
            nextlat_head_parameters nextlat_lr nextlat_horizon
            nextlat_projection_factor nextlat_draft_length train_nextlat
            nextlat_rollout nextlat_kl_tokens nextlat_mse_coefficient
            nextlat_kl_coefficient
            """,
        ),
        (
            "config",
            "configuration_rollout",
            """
            rollout_batch_rows prompts_per_rollout samples_per_prompt
            prompt_tokens max_new_tokens temperature top_p top_k compile_rollout
            fast_rollout min_rollout_tokens_per_second thinking
            """,
        ),
        (
            "config",
            "configuration_optimization",
            """
            steps value_warmup_steps actor_lr critic_lr ppo_epochs clip_low
            clip_high positive_coefficient value_coefficient replay_token_budget
            replay_max_trajectories logit_chunk_tokens
            """,
        ),
        (
            "config",
            "configuration_runtime",
            """
            device_telemetry device_telemetry_interval_ms device_power_floor
            checkpoint_interval_seconds rollout_only
            gate_min_positive_trajectories gate_min_positive_groups
            gate_max_truncation_fraction seed
            """,
        ),
        (
            "rollout",
            "rollout_quality",
            """
            trajectories prompt_groups positive_trajectories positive_groups
            mixed_groups accuracy truncation_fraction response_length_mean
            response_length_p95 response_length_max
            """,
        ),
        (
            "rollout",
            "rollout_performance",
            """
            generated_tokens rollout_seconds rollout_tokens_per_second
            scheduled_rollout_tokens_per_second prefill_seconds decode_seconds
            decode_tokens_per_second target_decode_calls target_decode_positions
            scheduled_decode_tokens_per_second target_positions_per_decode_call
            """,
        ),
        (
            "rollout",
            "rollout_sampling",
            """
            sampling_scanned_vocabulary sampling_candidate_support
            sampling_full_policy_mass_lower_bound
            sampling_conditional_mass_lower_bound
            """,
        ),
        (
            "train",
            "auxiliary_nextlat",
            """
            nextlat_loss nextlat_smooth_l1 nextlat_categorical_kl
            nextlat_transitions nextlat_grad_norm nextlat_optimizer_steps
            """,
        ),
        (
            "rollout_live",
            "system_live",
            """
            decode_steps batch_rows prompt_groups scheduled_tokens wall_seconds
            scheduled_tokens_per_second peak_vram_bytes
            device_utilization_gpu_percent device_utilization_memory_percent
            device_power_draw_watts device_clocks_sm_mhz device_memory_used_mib
            """,
        ),
        (
            "rollout",
            "system_rollout",
            """
            device_power_draw_watts_mean device_power_draw_watts_min
            device_power_draw_watts_max device_seconds_below_power_floor
            device_clocks_sm_mhz_mean device_clocks_sm_mhz_min
            device_utilization_gpu_percent_mean peak_vram_bytes
            """,
        ),
        (
            "rollout",
            "system_rollout_metadata",
            """
            device_power_draw_watts_period_ms device_power_draw_watts_readings
            device_clocks_sm_mhz_period_ms device_clocks_sm_mhz_readings
            device_utilization_gpu_percent_period_ms
            device_utilization_gpu_percent_readings
            """,
        ),
        (
            "train",
            "system_update",
            """
            device_power_draw_watts_mean device_power_draw_watts_min
            device_power_draw_watts_max device_seconds_below_power_floor
            device_clocks_sm_mhz_mean device_clocks_sm_mhz_min
            device_utilization_gpu_percent_mean peak_vram_bytes
            """,
        ),
        (
            "train",
            "system_update_metadata",
            """
            device_power_draw_watts_period_ms device_power_draw_watts_readings
            device_clocks_sm_mhz_period_ms device_clocks_sm_mhz_readings
            device_utilization_gpu_percent_period_ms
            device_utilization_gpu_percent_readings
            """,
        ),
    )
    expected: dict[str, str] = {}
    for source_prefix, destination_prefix, suffixes in groups:
        expected.update(
            _expected_prefix_mapping(
                source_prefix,
                destination_prefix,
                suffixes,
            )
        )
    expected.update(
        {
            "rollout/replay_storage_bytes": "replay/storage_bytes",
            "rollout/replay_bytes_per_token": "replay/bytes_per_token",
            "train/replay_microbatches": "replay/microbatches",
            "train/replay_actions": "replay/actions",
            "train/policy_loss": "actor/policy_loss",
            "train/positive_lm_loss": "actor/positive_lm_loss",
            "train/approximate_kl": "actor/approximate_kl",
            "train/sampled_forward_kl": "actor/sampled_forward_kl",
            "train/clip_fraction": "actor/clip_fraction",
            "train/ratio_mean": "actor/ratio_mean",
            "train/ratio_std": "actor/ratio_std",
            "train/actor_grad_norm": "actor/grad_norm",
            "train/value_loss": "critic/loss",
            "train/value_mean": "critic/prediction_mean",
            "train/value_target_mean": "critic/target_mean",
            "train/explained_variance": "critic/explained_variance",
            "train/critic_grad_norm": "critic/grad_norm",
            "train/loss": "optimization/total_loss",
            "train/update_seconds": "optimization/update_seconds",
        }
    )
    return expected




def test_audited_schema_is_total_unique_and_category_capped() -> None:
    assert len(LEGACY_SCALAR_TAG_MAP) == EXPECTED_LEGACY_SCALAR_TAGS == 143
    assert len(LEGACY_TEXT_TAG_MAP) == EXPECTED_LEGACY_TEXT_TAGS == 3
    assert len(LEGACY_TAG_MAP) == 146
    assert len(set(LEGACY_TAG_MAP.values())) == len(LEGACY_TAG_MAP)
    assert LEGACY_SCALAR_TAG_MAP == _expected_legacy_scalar_map()
    assert LEGACY_TEXT_TAG_MAP == {
        "config/arguments": "samples/configuration",
        "rollout_samples/correct": "samples/rollout_correct",
        "rollout_samples/incorrect": "samples/rollout_incorrect",
    }

    counts = validate_scalar_category_cap()
    assert max(counts.values()) == MAX_SCALAR_TAGS_PER_CATEGORY == 12
    assert counts["configuration_rollout"] == 12
    assert counts["configuration_optimization"] == 12
    assert counts["system_live"] == 12
    assert LEGACY_SCALAR_TAG_MAP["rollout/peak_vram_bytes"] == (
        "system_rollout/peak_vram_bytes"
    )
    assert "rollout/peak_vram_bytes" not in {
        source
        for source, destination in LEGACY_SCALAR_TAG_MAP.items()
        if destination == "rollout_performance/peak_vram_bytes"
    }

    oversized = {
        f"old/tag_{index}": f"too_large/tag_{index}" for index in range(13)
    }
    with pytest.raises(ValueError, match="too_large=13"):
        validate_scalar_category_cap(oversized)


def test_migration_is_lossless_ordered_and_hides_every_legacy_event_file(
    tmp_path: Path,
) -> None:
    source = _source_directory(tmp_path)
    before = {path.name: path.read_bytes() for path in _event_paths(source)}
    siblings_before = set(tmp_path.iterdir())

    dry_run = migration.migrate_tensorboard(source, dry_run=True)

    assert dry_run.dry_run is True
    assert dry_run.backup_directory is None
    assert {path.name: path.read_bytes() for path in _event_paths(source)} == before
    assert set(tmp_path.iterdir()) == siblings_before
    assert dry_run.summary_value_count == 8
    assert len(dry_run.event_files) == 2

    report = migration.migrate_tensorboard(source)

    assert report.dry_run is False
    backup = Path(report.backup_directory or "")
    assert backup.parent == source.parent
    assert backup.name.startswith("tensorboard.pre-migration-")
    assert not backup.is_relative_to(source)
    assert {path.name: path.read_bytes() for path in _event_paths(backup)} == before

    expected = _organized_source_events(backup)
    migrated = _events(source)
    assert migration._is_writer_boilerplate(migrated[0])
    actual = tuple(
        event.SerializeToString(deterministic=True) for event in migrated[1:]
    )
    assert actual == expected
    assert Counter(actual) == Counter(expected)

    migrated_values = [
        (event.step, event.wall_time, value)
        for event in migrated[1:]
        for value in event.summary.value
    ]
    duplicate_accuracy = [
        (step, wall_time, value.simple_value)
        for step, wall_time, value in migrated_values
        if value.tag == "rollout_quality/accuracy"
    ]
    assert duplicate_accuracy == [(7, 101.0, 0.25), (7, 102.0, 0.75)]
    assert sum(
        event.HasField("session_log") and event.session_log.status == SessionLog.START
        for event in migrated[1:]
    ) == 1
    assert any(
        len(event.summary.value) == 2
        and {value.tag for value in event.summary.value}
        == {"optimization/total_loss", "samples/rollout_correct/text_summary"}
        for event in migrated[1:]
    )
    text_values = {
        value.tag: value
        for _, _, value in migrated_values
        if value.metadata.plugin_data.plugin_name == "text"
    }
    assert set(text_values) == {
        "samples/configuration/text_summary",
        "samples/rollout_correct/text_summary",
        "samples/rollout_incorrect/text_summary",
    }
    assert text_values["samples/rollout_correct/text_summary"].tensor.string_val == [
        b"correct sample"
    ]

    migrated_tags = {value.tag for _, _, value in migrated_values}
    legacy_serialized_tags = set(LEGACY_SCALAR_TAG_MAP) | {
        f"{tag}/text_summary" for tag in LEGACY_TEXT_TAG_MAP
    }
    assert not migrated_tags & legacy_serialized_tags
    assert len(_event_paths(source)) == 1
    assert tuple(source.rglob("events.out.tfevents.*")) == _event_paths(source)
    assert not _all_accumulator_tags(source) & legacy_serialized_tags

    with pytest.raises(migration.AlreadyMigratedError, match="already contains"):
        migration.migrate_tensorboard(source, dry_run=True)


def test_failed_second_atomic_rename_restores_the_untouched_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_directory(tmp_path)
    before = {path.name: path.read_bytes() for path in _event_paths(source)}
    real_replace = migration._atomic_replace
    calls = 0

    def fail_install(source_path: Path, destination_path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected destination rename failure")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(migration, "_atomic_replace", fail_install)

    with pytest.raises(
        migration.TensorBoardMigrationError,
        match="cutover failed and was rolled back",
    ):
        migration.migrate_tensorboard(source)

    assert calls == 3
    assert {path.name: path.read_bytes() for path in _event_paths(source)} == before
    assert not tuple(tmp_path.glob("tensorboard.pre-migration-*"))
    assert not tuple(tmp_path.glob(".tensorboard.migrating-*"))


def test_source_hash_race_refuses_cutover_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_directory(tmp_path)
    real_rewrite = migration._rewrite_events
    raced_bytes: dict[str, bytes] = {}

    def rewrite_then_change_source(
        inspection: migration._SourceInspection,
        staging_directory: Path,
    ) -> tuple[bytes, ...]:
        expected = real_rewrite(inspection, staging_directory)
        raced_path = _event_paths(source)[0]
        with raced_path.open("ab") as stream:
            stream.write(b"injected writer race")
        raced_bytes[raced_path.name] = raced_path.read_bytes()
        return expected

    monkeypatch.setattr(migration, "_rewrite_events", rewrite_then_change_source)

    with pytest.raises(migration.SourceChangedError, match="changed"):
        migration.migrate_tensorboard(source)

    assert raced_bytes
    for name, expected_bytes in raced_bytes.items():
        assert (source / name).read_bytes() == expected_bytes
    assert not tuple(tmp_path.glob("tensorboard.pre-migration-*"))
    assert not tuple(tmp_path.glob(".tensorboard.migrating-*"))


def test_staging_payload_mismatch_is_rejected_before_cutover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_directory(tmp_path)
    before = {path.name: path.read_bytes() for path in _event_paths(source)}
    real_rewrite = migration._rewrite_events

    def rewrite_then_corrupt_staging(
        inspection: migration._SourceInspection,
        staging_directory: Path,
    ) -> tuple[bytes, ...]:
        expected = real_rewrite(inspection, staging_directory)
        staged = list(_events(staging_directory)[1:])
        corrupted = next(
            value
            for event in staged
            for value in event.summary.value
            if value.tag == "configuration_optimization/steps"
        )
        corrupted.simple_value += 1.0
        for event_path in _event_paths(staging_directory):
            event_path.unlink()
        _write_event_file(staging_directory, staged)
        return expected

    monkeypatch.setattr(migration, "_rewrite_events", rewrite_then_corrupt_staging)

    with pytest.raises(migration.StagingValidationError, match="validation failed"):
        migration.migrate_tensorboard(source)

    assert {path.name: path.read_bytes() for path in _event_paths(source)} == before
    assert not tuple(tmp_path.glob("tensorboard.pre-migration-*"))
    assert not tuple(tmp_path.glob(".tensorboard.migrating-*"))


def test_post_install_failure_moves_staging_aside_before_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_directory(tmp_path)
    before = {path.name: path.read_bytes() for path in _event_paths(source)}
    real_fsync_directory = migration._fsync_directory
    parent_fsyncs = 0

    def fail_after_install(directory: Path) -> None:
        nonlocal parent_fsyncs
        if directory == tmp_path:
            parent_fsyncs += 1
            if parent_fsyncs == 2:
                raise OSError("injected post-install fsync failure")
        real_fsync_directory(directory)

    monkeypatch.setattr(migration, "_fsync_directory", fail_after_install)

    with pytest.raises(
        migration.TensorBoardMigrationError,
        match="cutover failed and was rolled back",
    ):
        migration.migrate_tensorboard(source)

    assert parent_fsyncs == 3
    assert {path.name: path.read_bytes() for path in _event_paths(source)} == before
    assert not tuple(tmp_path.glob("tensorboard.pre-migration-*"))
    assert not tuple(tmp_path.glob(".tensorboard.migrating-*"))
