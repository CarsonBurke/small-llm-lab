#!/usr/bin/env python3
"""Losslessly reorganize a stopped MiniCPM TensorBoard event directory."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from typing import NoReturn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from postraining.minicpm_tensorboard_schema import (
    LEGACY_SCALAR_TAG_MAP,
    LEGACY_TEXT_TAG_MAP,
    TEXT_SUMMARY_SUFFIX,
    organized_tag,
    validate_scalar_category_cap,
)

_EVENT_FILE_PREFIX = "events.out.tfevents."
_WRITER_FILE_VERSION = "brain.Event:2"
_HASH_CHUNK_BYTES = 1 << 20


class TensorBoardMigrationError(RuntimeError):
    """Base exception for a refused or failed migration."""


class AlreadyMigratedError(TensorBoardMigrationError):
    """Raised when the supplied directory contains organized tags."""


class SourceChangedError(TensorBoardMigrationError):
    """Raised when a source event file changes during migration."""


class StagingValidationError(TensorBoardMigrationError):
    """Raised when staged events are not an exact renamed clone."""


class CutoverRollbackError(TensorBoardMigrationError):
    """Raised when an atomic cutover fails and cannot be fully rolled back."""


@dataclass(frozen=True)
class EventFileSnapshot:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class MigrationReport:
    source_directory: str
    dry_run: bool
    event_files: tuple[EventFileSnapshot, ...]
    event_count: int
    summary_value_count: int
    legacy_tags: tuple[str, ...]
    organized_tags: tuple[str, ...]
    backup_directory: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _SourceInspection:
    directory: Path
    event_files: tuple[EventFileSnapshot, ...]
    event_count: int
    summary_value_count: int
    legacy_tags: frozenset[str]
    organized_tags: frozenset[str]

    def report(
        self,
        *,
        dry_run: bool,
        backup_directory: Path | None = None,
    ) -> MigrationReport:
        return MigrationReport(
            source_directory=str(self.directory),
            dry_run=dry_run,
            event_files=self.event_files,
            event_count=self.event_count,
            summary_value_count=self.summary_value_count,
            legacy_tags=tuple(sorted(self.legacy_tags)),
            organized_tags=tuple(sorted(self.organized_tags)),
            backup_directory=(
                str(backup_directory) if backup_directory is not None else None
            ),
        )


def _event_file_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.removeprefix(_EVENT_FILE_PREFIX)
    timestamp, _, _ = suffix.partition(".")
    try:
        return int(timestamp), path.name
    except ValueError:
        return sys.maxsize, path.name


def _event_files(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir():
        raise TensorBoardMigrationError(
            f"TensorBoard directory does not exist: {directory}"
        )
    files = tuple(
        sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.name.startswith(_EVENT_FILE_PREFIX)
            ),
            key=_event_file_sort_key,
        )
    )
    if not files:
        raise TensorBoardMigrationError(
            f"no {_EVENT_FILE_PREFIX}* files found directly in {directory}"
        )
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(directory: Path) -> tuple[EventFileSnapshot, ...]:
    return tuple(
        EventFileSnapshot(path.name, path.stat().st_size, _sha256(path))
        for path in _event_files(directory)
    )


def _snapshot_difference(
    expected: Sequence[EventFileSnapshot],
    actual: Sequence[EventFileSnapshot],
) -> str:
    expected_by_name = {item.name: item for item in expected}
    actual_by_name = {item.name: item for item in actual}
    added = sorted(actual_by_name.keys() - expected_by_name.keys())
    removed = sorted(expected_by_name.keys() - actual_by_name.keys())
    changed = sorted(
        name
        for name in expected_by_name.keys() & actual_by_name.keys()
        if expected_by_name[name] != actual_by_name[name]
    )
    details = []
    if added:
        details.append(f"added={added!r}")
    if removed:
        details.append(f"removed={removed!r}")
    if changed:
        details.append(f"changed={changed!r}")
    return ", ".join(details) or "file ordering changed"


def _assert_snapshot(
    directory: Path,
    expected: Sequence[EventFileSnapshot],
) -> None:
    actual = _snapshot(directory)
    if tuple(expected) != actual:
        raise SourceChangedError(
            "source TensorBoard event files changed during migration: "
            + _snapshot_difference(expected, actual)
        )


def _parse_event(raw: bytes, path: Path) -> Event:
    event = Event()
    try:
        event.ParseFromString(raw)
    except Exception as error:
        raise TensorBoardMigrationError(
            f"could not parse TensorBoard event from {path}: {error}"
        ) from error
    return event


def _load_events(path: Path) -> Iterator[Event]:
    for raw in RawEventFileLoader(str(path)).Load():
        yield _parse_event(raw, path)


def _serialized_destination_tags() -> frozenset[str]:
    return frozenset(LEGACY_SCALAR_TAG_MAP.values()) | frozenset(
        f"{tag}{TEXT_SUMMARY_SUFFIX}" for tag in LEGACY_TEXT_TAG_MAP.values()
    ) | frozenset(LEGACY_TEXT_TAG_MAP.values())


_DESTINATION_TAGS = _serialized_destination_tags()


def _translate_source_tag(tag: str) -> str:
    if tag in _DESTINATION_TAGS:
        raise AlreadyMigratedError(
            f"TensorBoard directory already contains organized tag {tag!r}"
        )
    try:
        return organized_tag(tag)
    except KeyError as error:
        raise TensorBoardMigrationError(
            f"no audited MiniCPM TensorBoard mapping for source tag {tag!r}"
        ) from error


def _inspect_source(directory: Path) -> _SourceInspection:
    directory = directory.expanduser().resolve()
    validate_scalar_category_cap()
    snapshots = _snapshot(directory)
    event_count = 0
    value_count = 0
    source_tags: set[str] = set()
    destination_tags: set[str] = set()
    for snapshot in snapshots:
        path = directory / snapshot.name
        for event in _load_events(path):
            event_count += 1
            for value in event.summary.value:
                source_tags.add(value.tag)
                destination_tags.add(_translate_source_tag(value.tag))
                value_count += 1
    if not source_tags:
        raise TensorBoardMigrationError(
            f"no tagged TensorBoard summary values found in {directory}"
        )
    _assert_snapshot(directory, snapshots)
    return _SourceInspection(
        directory=directory,
        event_files=snapshots,
        event_count=event_count,
        summary_value_count=value_count,
        legacy_tags=frozenset(source_tags),
        organized_tags=frozenset(destination_tags),
    )


def _clone_with_organized_tags(event: Event) -> Event:
    cloned = Event()
    cloned.CopyFrom(event)
    for value in cloned.summary.value:
        value.tag = _translate_source_tag(value.tag)
    return cloned


def _normalized_event(event: Event) -> bytes:
    return event.SerializeToString(deterministic=True)


def _summary_value_records(serialized_events: Iterable[bytes]) -> Counter[tuple]:
    records: Counter[tuple] = Counter()
    for serialized in serialized_events:
        event = _parse_event(serialized, Path("<validation>"))
        wall_time = struct.pack(">d", event.wall_time)
        for value in event.summary.value:
            payload = Summary.Value()
            payload.CopyFrom(value)
            payload.ClearField("tag")
            records[
                (
                    value.tag,
                    event.step,
                    wall_time,
                    payload.SerializeToString(deterministic=True),
                )
            ] += 1
    return records


def _rewrite_events(
    inspection: _SourceInspection,
    staging_directory: Path,
) -> tuple[bytes, ...]:
    expected: list[bytes] = []
    writer = EventFileWriter(str(staging_directory))
    try:
        for snapshot in inspection.event_files:
            source_path = inspection.directory / snapshot.name
            for event in _load_events(source_path):
                cloned = _clone_with_organized_tags(event)
                expected.append(_normalized_event(cloned))
                writer.add_event(cloned)
    finally:
        writer.close()
    return tuple(expected)


def _is_writer_boilerplate(event: Event) -> bool:
    return (
        event.file_version == _WRITER_FILE_VERSION
        and not event.HasField("summary")
        and event.step == 0
    )


def _staged_events(staging_directory: Path) -> tuple[Event, ...]:
    files = _event_files(staging_directory)
    if len(files) != 1:
        raise StagingValidationError(
            f"staging writer produced {len(files)} event files instead of one"
        )
    return tuple(_load_events(files[0]))


def _first_difference(expected: Sequence[bytes], actual: Sequence[bytes]) -> int:
    for index, (expected_event, actual_event) in enumerate(zip(expected, actual)):
        if expected_event != actual_event:
            return index
    return min(len(expected), len(actual))


def _validate_staging(
    inspection: _SourceInspection,
    staging_directory: Path,
    expected_events: Sequence[bytes],
) -> None:
    staged = _staged_events(staging_directory)
    if not staged or not _is_writer_boilerplate(staged[0]):
        raise StagingValidationError(
            "staged event file is missing EventFileWriter file-version boilerplate"
        )
    actual_events = tuple(_normalized_event(event) for event in staged[1:])
    expected_tuple = tuple(expected_events)
    if actual_events != expected_tuple:
        expected_multiset = Counter(expected_tuple)
        actual_multiset = Counter(actual_events)
        if expected_multiset != actual_multiset:
            detail = (
                f"event multiset differs (expected {len(expected_tuple)}, "
                f"found {len(actual_events)})"
            )
        else:
            detail = "event order differs despite equal event multisets"
        index = _first_difference(expected_tuple, actual_events)
        raise StagingValidationError(
            f"staged protobuf validation failed at event {index}: {detail}"
        )
    expected_values = _summary_value_records(expected_tuple)
    actual_values = _summary_value_records(actual_events)
    if expected_values != actual_values:
        raise StagingValidationError(
            "staged summary value payload/cardinality multiset differs from source"
        )
    actual_tags = {
        value.tag for event in staged[1:] for value in event.summary.value
    }
    if actual_tags != inspection.organized_tags:
        raise StagingValidationError(
            "staged organized tag set differs from the audited source translation"
        )
    legacy_serialized_tags = set(LEGACY_SCALAR_TAG_MAP) | {
        f"{tag}{TEXT_SUMMARY_SUFFIX}" for tag in LEGACY_TEXT_TAG_MAP
    } | set(LEGACY_TEXT_TAG_MAP)
    surviving = sorted(actual_tags & legacy_serialized_tags)
    if surviving:
        raise StagingValidationError(
            f"legacy tags survived staged migration: {surviving!r}"
        )
    if sum(expected_values.values()) != inspection.summary_value_count:
        raise StagingValidationError(
            "staged summary value cardinality differs from the source inspection"
        )


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_staging(staging_directory: Path) -> None:
    for path in _event_files(staging_directory):
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    _fsync_directory(staging_directory)


def _atomic_replace(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _backup_path(directory: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    candidate = directory.with_name(
        f"{directory.name}.pre-migration-{timestamp}"
    )
    if candidate.exists():
        raise TensorBoardMigrationError(
            f"refusing to overwrite migration backup: {candidate}"
        )
    return candidate


def _remove_staging(staging_directory: Path) -> None:
    if staging_directory.exists():
        shutil.rmtree(staging_directory)


def _raise_rollback_failure(
    original_error: BaseException,
    rollback_error: BaseException,
    *,
    directory: Path,
    backup: Path,
    staging: Path,
) -> NoReturn:
    raise CutoverRollbackError(
        "TensorBoard cutover failed and rollback was incomplete; preserve all "
        f"paths for recovery: destination={directory}, backup={backup}, "
        f"staging={staging}; cutover error={original_error!r}; "
        f"rollback error={rollback_error!r}"
    ) from rollback_error


def _cutover(
    inspection: _SourceInspection,
    staging_directory: Path,
) -> Path:
    directory = inspection.directory
    parent = directory.parent
    backup = _backup_path(directory)
    source_moved = False
    staged_installed = False
    try:
        _assert_snapshot(directory, inspection.event_files)
        _atomic_replace(directory, backup)
        source_moved = True
        _fsync_directory(parent)
        _assert_snapshot(backup, inspection.event_files)
        _atomic_replace(staging_directory, directory)
        staged_installed = True
        _fsync_directory(parent)
        _assert_snapshot(backup, inspection.event_files)
    except BaseException as error:
        try:
            if staged_installed:
                _atomic_replace(directory, staging_directory)
                staged_installed = False
            if source_moved:
                _atomic_replace(backup, directory)
                source_moved = False
            _fsync_directory(parent)
        except BaseException as rollback_error:
            _raise_rollback_failure(
                error,
                rollback_error,
                directory=directory,
                backup=backup,
                staging=staging_directory,
            )
        _remove_staging(staging_directory)
        if isinstance(error, TensorBoardMigrationError):
            raise
        raise TensorBoardMigrationError(
            f"atomic TensorBoard cutover failed and was rolled back: {error}"
        ) from error
    return backup


def migrate_tensorboard(
    tensorboard_directory: str | Path,
    *,
    dry_run: bool = False,
) -> MigrationReport:
    """Inspect or atomically migrate one stopped TensorBoard directory."""
    inspection = _inspect_source(Path(tensorboard_directory))
    if dry_run:
        return inspection.report(dry_run=True)

    prefix = f".{inspection.directory.name}.migrating-"
    staging_directory = Path(
        tempfile.mkdtemp(prefix=prefix, dir=inspection.directory.parent)
    )
    try:
        expected_events = _rewrite_events(inspection, staging_directory)
        _fsync_staging(staging_directory)
        _validate_staging(inspection, staging_directory, expected_events)
        _assert_snapshot(inspection.directory, inspection.event_files)
    except BaseException:
        _remove_staging(staging_directory)
        raise

    backup = _cutover(inspection, staging_directory)
    return inspection.report(dry_run=False, backup_directory=backup)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tensorboard_directory",
        type=Path,
        help="stopped run's TensorBoard directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="hash and validate the source without writing or renaming anything",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = migrate_tensorboard(
            args.tensorboard_directory,
            dry_run=args.dry_run,
        )
    except TensorBoardMigrationError as error:
        parser.error(str(error))
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
