"""Stream a trainer console log into metrics JSONL and TensorBoard.

Runs launched directly through mlq (rather than scripts/ablation.py) write
only their ``logs/<run_id>.txt``; nothing feeds ``tb_logs/``. This bridge
replays the existing log from the beginning and then follows it, routing
every line through the ablation parser into the same ``MetricsWriter`` the
ablation runner uses, so the run appears in TensorBoard (``tb_logs/<name>``)
and ``ablation_results/<name>/metrics.jsonl`` exactly as an ablation-launched
run would.

Usage:
    .venv/bin/python -m scripts.tb_log_bridge logs/my_run.txt --name my_run

The bridge exits once it has seen the trainer's final ``saved checkpoint:``
line and drained the log, or after ``--idle-exit`` seconds without new bytes
(a crashed run never prints the checkpoint line).
"""

import argparse
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ablation import REPO_ROOT, MetricsWriter, parse_log_line

CHECKPOINT_MARKER = "saved checkpoint: "


def read_new_lines(
    path: Path, position: int, partial_line: str
) -> tuple[list[str], int, str]:
    try:
        size = path.stat().st_size
    except OSError:
        return [], position, partial_line
    if size < position:
        # The log was truncated or replaced; start over.
        position = 0
        partial_line = ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(position)
            chunk = f.read()
            position = f.tell()
    except OSError:
        return [], position, partial_line
    if not chunk:
        return [], position, partial_line
    chunk = partial_line + chunk
    lines = chunk.splitlines(keepends=True)
    if lines and not lines[-1].endswith(("\n", "\r")):
        partial_line = lines.pop()
    else:
        partial_line = ""
    return [line.rstrip("\r\n") for line in lines], position, partial_line


def bridge(log_path: Path, name: str, idle_exit_s: float, poll_s: float) -> None:
    metrics_dir = REPO_ROOT / "ablation_results" / name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    writer = MetricsWriter(metrics_dir / "metrics.jsonl", name)
    position = 0
    partial_line = ""
    finished = False
    last_progress = time.monotonic()
    print(f"Bridging {log_path} -> tb_logs/{name}", flush=True)
    try:
        while True:
            lines, position, partial_line = read_new_lines(
                log_path, position, partial_line
            )
            if lines:
                last_progress = time.monotonic()
            for line in lines:
                # Line-start match only: the trainer logs its own source code
                # first, and the print0 call that emits this marker appears
                # there as an indented line containing the same substring.
                if line.startswith(CHECKPOINT_MARKER):
                    finished = True
                entry = parse_log_line(line)
                if entry is not None:
                    writer.write_entry(entry)
            if finished and not lines and not partial_line:
                print(f"Run complete; bridge for {name} exiting.", flush=True)
                return
            if time.monotonic() - last_progress > idle_exit_s:
                print(
                    f"No new log output for {idle_exit_s:.0f}s; bridge for "
                    f"{name} exiting without a checkpoint marker.",
                    flush=True,
                )
                return
            time.sleep(poll_s)
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path, help="trainer console log to follow")
    parser.add_argument("--name", required=True, help="TensorBoard run name")
    parser.add_argument(
        "--idle-exit",
        type=float,
        default=7200.0,
        help="exit after this many seconds without new log bytes",
    )
    parser.add_argument("--poll", type=float, default=2.0)
    args = parser.parse_args()
    bridge(args.log_path, args.name, args.idle_exit, args.poll)


if __name__ == "__main__":
    main()
