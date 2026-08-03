"""Run scripts/ablation.py, then exit with the TRAINING script's recorded returncode.

scripts/ablation.py always exits 0 once it has written result.json, even when the
training subprocess failed — which silently satisfies mlq --after-success
dependency chains.  This wrapper re-reads result.json and propagates the
recorded returncode so "after-success" means the training actually trained.
Usage: identical arguments to scripts/ablation.py (must include --name).
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> int:
    args = sys.argv[1:]
    if "--name" not in args:
        print("run_ablation_gated: --name is required", file=sys.stderr)
        return 2
    name = args[args.index("--name") + 1]
    rc = subprocess.call([sys.executable, "scripts/ablation.py", *args], cwd=ROOT)
    if rc:
        return rc
    result_path = ROOT / "ablation_results" / name / "result.json"
    try:
        result = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"run_ablation_gated: cannot read {result_path}: {exc}", file=sys.stderr)
        return 3
    inner = result.get("returncode")
    if inner != 0:
        print(f"run_ablation_gated: training returncode was {inner}", file=sys.stderr)
        return inner if isinstance(inner, int) and 0 < inner < 128 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
