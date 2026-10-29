#!/usr/bin/env python3
"""Read-only host and per-attempt /proc memory sampling; never starts workloads."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def proc_identity(text: str) -> tuple[int, int, int]:
    # comm may contain spaces or ')'; fields after its final ')' start at state.
    fields = text[text.rfind(")") + 2 :].split()
    return int(fields[1]), int(fields[2]), int(fields[19])


def memory_fields(path: Path) -> dict[str, int]:
    result = {}
    for line in path.read_text().splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            result[key] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1] == "kB" else 1)
    return result


def sample_processes(roots: dict[int, int], pgid: int | None) -> list[dict]:
    identities = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                identities[int(path.name)] = proc_identity((path / "stat").read_text())
            except (OSError, ValueError, IndexError):
                pass  # Processes may disappear between reads.
    selected = {pid for pid, start in roots.items() if pid in identities and identities[pid][2] == start}
    # A stale/reused leader must never cause an unrelated process-group sample.
    if selected and pgid is not None:
        selected.update(pid for pid, (_, group, _) in identities.items() if group == pgid)
    while True:
        children = {pid for pid, (parent, _, _) in identities.items() if parent in selected}
        if children <= selected:
            break
        selected.update(children)
    rows = []
    for pid in sorted(selected):
        path = Path("/proc") / str(pid)
        try:
            status = memory_fields(path / "status")
            row = {"pid": pid, "ppid": identities[pid][0], "start_ticks": identities[pid][2],
                   "rss_bytes": status.get("VmRSS", 0), "swap_bytes": status.get("VmSwap", 0),
                   "command": (path / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip(),
                   "cgroup": (path / "cgroup").read_text().strip()}
            try:
                smaps = memory_fields(path / "smaps_rollup")
                row.update(pss_bytes=smaps.get("Pss"), swap_pss_bytes=smaps.get("SwapPss"))
            except OSError:
                row["pss_unavailable"] = True
            if proc_identity((path / "stat").read_text())[2] == row["start_ticks"]:
                rows.append(row)
        except (OSError, ValueError, IndexError):
            pass
    return rows


def sample(job_id: int, state_root: Path) -> dict:
    row = {"time_unix": time.time(), "job_id": job_id, "host": memory_fields(Path("/proc/meminfo"))}
    result = subprocess.run(["mlq", "show", str(job_id), "--json"], capture_output=True, text=True, timeout=10)
    if result.returncode:
        row["queue_error"] = result.stderr.strip()
        return row
    job = json.loads(result.stdout)
    row["job_state"] = job["state"]
    if not job.get("attempts"):
        return row
    attempt = max(job["attempts"], key=lambda item: item["number"])
    row.update(attempt_id=attempt["id"], attempt_state=attempt["state"])
    folder = state_root / "attempts" / str(attempt["id"])
    roots = {}
    pgid = None
    for filename, pid_key, start_key in [("identity.json", "runner_pid", "runner_start_time"),
                                          ("exec.json", "pid", "start_time")]:
        try:
            payload = json.loads((folder / filename).read_text())
            # /proc start_ticks alone may repeat after reboot.
            if payload["boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
                continue
            roots[payload[pid_key]] = payload[start_key]
            if filename == "exec.json":
                pgid = payload["pgid"]
        except (OSError, ValueError, KeyError):
            continue
    rows = sample_processes(roots, pgid)
    row["processes"] = rows
    row["totals"] = {key: sum(process.get(key, 0) or 0 for process in rows)
                     for key in ["rss_bytes", "swap_bytes", "pss_bytes", "swap_pss_bytes"]}
    row["totals"]["pss_complete"] = all("pss_bytes" in process for process in rows)
    # RSS double-counts shared pages; PSS is the preferable physical-memory estimate.
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=5)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--state-root", type=Path, default=Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state"))) / "mlqueue")
    args = parser.parse_args()
    if not 1 <= args.interval_seconds <= 60:
        parser.error("interval must be between 1 and 60 seconds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", buffering=1) as output:
        while True:
            try:
                row = sample(args.job, args.state_root)
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                row = {"time_unix": time.time(), "job_id": args.job, "monitor_error": str(error)}
            output.write(json.dumps(row, allow_nan=False) + "\n")
            if args.once or row.get("job_state") in {"succeeded", "failed", "lost", "cancelled"}:
                break
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
