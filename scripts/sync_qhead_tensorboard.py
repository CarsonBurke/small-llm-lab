"""Backfill missing qhead charts from canonical metrics, then follow append-only JSONL.

The original qhead critic is fitted to lambda-one (Monte Carlo) returns. Its
online-epoch V diagnostics therefore provide the MC aliases below; the separate
Q residual diagnostics never enter those aliases. This adds only missing tags,
without changing training or the original events.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import struct
import subprocess
import tarfile
from pathlib import Path
import time

SCHEMA = "qhead_canonical_metric_aliases/v1"
EVENT_SUFFIX = ".qhead_compat"
VERIFIED_CORE_SHA256 = "b3e3f8285bd49199ec5badbca1bc6f068b91c4b51f052f9f539ca0df41e0f453"
VERIFIED_TRAINER_SHA256 = "e30e0170f172e3796c36e201bbeadb31ef5abe222d11d368316bc21e5589438e"
VALUE_ALIASES = {
    "value/mc_explained_variance": "value/online_epoch_explained_variance",
    "value/mc_target_mean": "value/target_mean",
    "value/mc_residual_mean": "value/residual_mean",
}
SOURCE_METRICS = ("think_gate_zeroed_correct_fraction", "think_tokens_mean")
MAX_LINE_BYTES = 4 * 1024 * 1024


class MetricsFileChanged(RuntimeError):
    """Canonical metrics were replaced or rewritten; operator must rebuild."""


class ScalarEventWriter:
    """Write scalar protobuf events without importing the training framework."""

    def __init__(self, directory: Path):
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        self.writer = EventFileWriter(
            str(directory), filename_suffix=EVENT_SUFFIX, flush_secs=5
        )

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        self.writer.add_event(Event(
            wall_time=time.time(), step=step,
            summary=Summary(value=[Summary.Value(tag=tag, simple_value=value)]),
        ))

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def scalar_mapping(record: dict) -> dict[str, float]:
    """Map only canonical measured values, preserving the record's actor step."""
    mapped = {}
    if record.get("type") == "train":
        dashboard = record.get("dashboard") or {}
        for destination, source in VALUE_ALIASES.items():
            if source in dashboard:
                mapped[destination] = dashboard[source]
    if record.get("type") == "rollout":
        for name, metrics in (record.get("source_metrics") or {}).items():
            for metric in SOURCE_METRICS:
                if metric in metrics:
                    mapped[f"source/{name}/{metric}"] = metrics[metric]
    return {
        tag: float(value) for tag, value in mapped.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value)
    }


def anchor(stream, position: int) -> str:
    stream.seek(max(0, position - 256))
    return hashlib.sha256(stream.read(min(position, 256))).hexdigest()


def event_points(directory: Path) -> dict[tuple[str, int], float]:
    """Recover emitted points, including a flush that preceded a state-file crash."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    points = {}
    for path in sorted(directory.glob(f"events.out.tfevents.*{EVENT_SUFFIX}")):
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload()
        for tag in accumulator.Tags()["scalars"]:
            for item in accumulator.Scalars(tag):
                points[tag, item.step] = item.value
    return points


def source_provenance(run: Path) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    critic = manifest.get("critic") or {}
    if critic.get("critic_schema") != "scalar_zero_init_linear_unclipped_mse/v1":
        raise ValueError("this alias bridge requires the original scalar V critic schema")
    source = manifest.get("source_provenance") or {}
    archive = run / source["source_archive"]
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != source["source_archive_sha256"]:
            raise ValueError("run source archive checksum mismatch")
    with tarfile.open(archive) as snapshot:
        core = snapshot.extractfile("postraining/core.py")
        if core is None or hashlib.sha256(core.read()).hexdigest() != VERIFIED_CORE_SHA256:
            raise ValueError("MC aliases require the verified original lambda-one return implementation")
        trainer = snapshot.extractfile("postraining/train_latent_vapo.py")
        if trainer is None or hashlib.sha256(trainer.read()).hexdigest() != VERIFIED_TRAINER_SHA256:
            raise ValueError("MC aliases require the verified original trainer target semantics")
    return {
        "schema": SCHEMA,
        "verified_core_sha256": VERIFIED_CORE_SHA256,
        "verified_trainer_sha256": VERIFIED_TRAINER_SHA256,
        "run": str(run.resolve()),
        "source_archive": (manifest.get("source_provenance") or {}).get("source_archive"),
        "source_archive_sha256": (manifest.get("source_provenance") or {}).get("source_archive_sha256"),
        "value_aliases": VALUE_ALIASES,
        "value_target_semantics": "V diagnostics against lambda-one undiscounted episodic returns; actor lambda does not change these targets",
        "verified_implementation": "original qhead core.generalized_advantage_and_return_targets and train_latent_vapo.aggregate_value_diagnostics/actor_tensorboard_metrics",
        "source_metrics": list(SOURCE_METRICS),
        "step_semantics": "canonical record.step, no offset",
        "unavailable_metrics": ["source/*/length_reward_covariance"],
    }


class MetricSync:
    def __init__(self, run: Path, writer, *, emitted=None):
        self.run = run.resolve()
        self.metrics = self.run / "metrics.jsonl"
        self.state_path = self.run / "tensorboard_sync.state.json"
        self.writer = writer
        self.emitted = emitted if emitted is not None else event_points(self.run / "tensorboard")
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else None
        if self.state and (self.state.get("schema") != SCHEMA or self.state.get("metrics") != str(self.metrics)):
            raise ValueError("sidecar state belongs to a different source/schema")

    def sync(self, max_lines: int = 500) -> int:
        if max_lines < 1:
            raise ValueError("max_lines must be positive")
        consumed = 0
        with self.metrics.open("rb") as stream:
            metadata = os.fstat(stream.fileno())
            position = (self.state or {}).get("offset", 0)
            if self.state:
                same_file = [metadata.st_dev, metadata.st_ino] == self.state["file_identity"]
                if not same_file or metadata.st_size < position or anchor(stream, position) != self.state["anchor"]:
                    raise MetricsFileChanged("metrics were replaced/truncated/rewritten; stop this sidecar and run --rebuild --once")
            stream.seek(position)
            for _ in range(max_lines):
                line_start = stream.tell()
                line = stream.readline(MAX_LINE_BYTES + 1)
                if len(line) > MAX_LINE_BYTES:
                    raise ValueError(f"metrics line at byte {line_start} exceeds {MAX_LINE_BYTES} bytes")
                if not line or not line.endswith(b"\n"):
                    # Partial writes remain unread until the newline is appended.
                    stream.seek(line_start)
                    break
                record = json.loads(line)
                step = record.get("step")
                mapped = scalar_mapping(record)
                if mapped and (not isinstance(step, int) or isinstance(step, bool) or step < 0):
                    raise ValueError(f"mapped metrics line at byte {line_start} has invalid actor step")
                for tag, value in mapped.items():
                    # SummaryWriter stores float32 scalars. Compare that representation
                    # so restart recovery does not duplicate unchanged float64 inputs.
                    stored_value = struct.unpack("f", struct.pack("f", value))[0]
                    if self.emitted.get((tag, step)) != stored_value:
                        self.writer.add_scalar(tag, value, step)
                        self.emitted[tag, step] = stored_value
                consumed += 1
            position = stream.tell()
            new_state = {
                "schema": SCHEMA, "metrics": str(self.metrics), "offset": position,
                "file_identity": [metadata.st_dev, metadata.st_ino],
                "anchor": anchor(stream, position),
            }
        # Flush first. A crash before committing offset is reconciled from events
        # on restart, making replay idempotent across that transaction window.
        self.writer.flush()
        atomic_json(self.state_path, new_state)
        self.state = new_state
        return consumed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="canonical post-training run directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="drain complete existing lines and exit")
    mode.add_argument("--follow", action="store_true", help="follow append-only metrics (default)")
    parser.add_argument("--job", type=int, help="exit after this mlq job finishes and complete metrics are drained")
    parser.add_argument("--rebuild", action="store_true", help="remove only this sidecar's events/state and rebuild from canonical metrics")
    parser.add_argument("--poll-seconds", type=float, default=5)
    args = parser.parse_args()
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("--poll-seconds must be finite and positive")
    if args.job is not None and (args.job < 1 or args.once):
        parser.error("--job must be positive and requires follow mode")
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    run = args.run.resolve()
    provenance = source_provenance(run)
    directory = run / "tensorboard"
    directory.mkdir(parents=True, exist_ok=True)
    with (run / "tensorboard_sync.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.rebuild:
            for path in directory.glob(f"events.out.tfevents.*{EVENT_SUFFIX}"):
                path.unlink()
            (run / "tensorboard_sync.state.json").unlink(missing_ok=True)
        atomic_json(run / "tensorboard_sync.provenance.json", provenance)
        writer = ScalarEventWriter(directory)
        follower = None
        try:
            if args.job is not None:
                follower = subprocess.Popen(
                    ["mlq", "follow", str(args.job), "--only-finish", "--no-hooks",
                     "--no-checks", "--timeout", "5h"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            sync = MetricSync(run, writer)
            print(f"Syncing canonical missing charts for {run.name}", flush=True)
            while True:
                consumed = sync.sync()
                if consumed:
                    print(f"Synced {consumed} records through byte {sync.state['offset']}", flush=True)
                elif args.once:
                    break
                elif follower is not None and follower.poll() is not None:
                    if follower.returncode == 124:
                        raise RuntimeError("mlq follow timed out; stopping sidecar without cancelling training")
                    print(f"mlq job {args.job} finished (exit {follower.returncode}); metrics drained", flush=True)
                    break
                else:
                    time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            pass
        finally:
            writer.close()
            if follower is not None and follower.poll() is None:
                follower.terminate()
                try:
                    follower.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    follower.kill()
                    follower.wait()


if __name__ == "__main__":
    main()
