"""
Ablation runner for parameter-golf.
Runs the baseline (or modified) train_gpt.py with different configs and
collects val_bpb at specified step checkpoints for comparison.

Usage:
    python3 -m scripts.ablation                              # Run baseline to 2000 steps
    python3 -m scripts.ablation --steps 1000                 # Run baseline to 1000 steps
    python3 -m scripts.ablation --sweep lr                   # Sweep learning rates
    python3 -m scripts.ablation --compare                    # Compare past runs
    python3 -m scripts.ablation --script ablations/sota_train_gpt.py --name sota_2k

Runs of at least 2,000 steps use the trusted 1,200-step warmdown by default;
short checkpoint runs stay at a flat learning rate. Override either behavior
explicitly with ``--env WARMDOWN_ITERS=...``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "ablation_results"
TB_DIR = REPO_ROOT / "tb_logs"
SANITIZED_ENV_PREFIXES = ("PURE_LEJEPA_", "LEJEPA_")
REFERENCE_ABLATION_STEPS = 2000
REFERENCE_WARMDOWN_ITERS = 1200


def parse_extra_metrics(extras: str) -> dict[str, float]:
    metrics = {}
    for key, value in re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*):([+-]?(?:\d+\.?\d*|\.\d+))", extras):
        if key == "train_time" or key.startswith("native_"):
            continue
        if key == "step_avg":
            key = "step_avg_ms"
        metrics[key] = float(value)
    return metrics


def parse_time_ms(text: str) -> float | None:
    match = re.search(r"train_time:\s*([+-]?(?:\d+\.?\d*|\.\d+))\s*(ms|m)?", text)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2) or "ms"
    if unit == "m":
        return value * 60_000
    return value


def parse_log_line(line: str) -> dict | None:
    """Parse a single log line into a metric dict."""
    graph_match = re.match(r"graph_stats\s+(.*)", line)
    if graph_match:
        metrics = parse_extra_metrics(graph_match.group(1))
        return {"type": "graph_stats", **metrics} if metrics else None

    churn_match = re.match(r"churn_stats\s+(.*)", line)
    if churn_match:
        metrics = parse_extra_metrics(churn_match.group(1))
        # stepless by design: emitted just before the val line it belongs to,
        # MetricsWriter folds it into that val entry
        return {"type": "churn_stats", **metrics} if metrics else None

    diag_match = re.match(r"token_view_diag:step:(\d+)/\d+\s+(.*)", line)
    if diag_match:
        entry = {
            "step": int(diag_match.group(1)),
            "type": "diag",
        }
        extras = diag_match.group(2).strip()
        if extras:
            entry.update(parse_extra_metrics(extras))
        return entry

    val_match = re.match(
        r"(?:step:)?(\d+)/\d+\s+val_loss:\s*([+-]?(?:\d+\.?\d*|\.\d+))\s+val_bpb:\s*([+-]?(?:\d+\.?\d*|\.\d+))\s*(.*)",
        line,
    )
    if val_match:
        extras = val_match.group(4).strip()
        entry = {
            "step": int(val_match.group(1)),
            "val_loss": float(val_match.group(2)),
            "val_bpb": float(val_match.group(3)),
            "train_time_ms": parse_time_ms(extras) or 0.0,
            "type": "val",
        }
        if extras:
            entry.update(parse_extra_metrics(extras))
        return entry
    train_match = re.match(
        r"(?:step:)?(\d+)/\d+\s+train_loss:\s*([+-]?(?:\d+\.?\d*|\.\d+))\s*(.*)",
        line,
    )
    if train_match:
        extras = train_match.group(3).strip()
        train_time_ms = parse_time_ms(extras)
        if train_time_ms is None:
            return None
        entry = {
            "step": int(train_match.group(1)),
            "train_loss": float(train_match.group(2)),
            "train_time_ms": train_time_ms,
            "type": "train",
        }
        if extras:
            entry.update(parse_extra_metrics(extras))
        return entry
    return None


class MetricsWriter:
    """Streams parsed metrics to JSONL and tensorboard."""

    def __init__(self, metrics_path: Path, name: str):
        self.metrics_path = metrics_path
        self.writer = SummaryWriter(log_dir=str(TB_DIR / name), max_queue=512, flush_secs=10)
        self.metrics_file = metrics_path.open("a", encoding="utf-8")
        self._prev_train_loss = None
        self._last_train_time_ms = None
        self._pending_churn: dict | None = None
        self._pending_graph: dict | None = None

    @staticmethod
    def _extra_scalar_tag(key: str) -> str:
        lejepa_keys = {
            "inv_loss",
            "sigreg_loss",
            "view_norm",
            "global_cos",
            "local_cos",
            "view_spread",
            "view_eff_rank",
            "global_mask_jaccard",
            "local_mask_jaccard",
            "global_pooled_cos",
            "local_pooled_cos",
            "pooled_view_eff_rank",
            "pooled_view_eff_rank_ratio",
            "sigreg_view_eff_rank",
            "sigreg_view_eff_rank_ratio",
            "latent_eff_rank",
            "latent_eff_rank_ratio",
            "next_pred_cos",
            "context_target_cos",
            "token_view_eff_rank",
            "token_view_eff_rank_ratio",
            "token_view_cos_off_mean",
            "token_view_dim_std_mean",
            "view_eff_rank_ratio",
            "sigreg_samples",
            "dirs_sum",
        }
        if key == "stage_id":
            return "stage/id"
        if key == "step_avg_ms":
            return "perf/step_avg_ms"
        if key.startswith("lejepa_"):
            return f"lejepa/{key.removeprefix('lejepa_')}"
        if key in lejepa_keys:
            return f"lejepa/{key}"
        if key.startswith("probe_"):
            return f"probe/{key.removeprefix('probe_')}"
        if re.match(r"(?:rewire|support|hnorm)_l\d+$", key):
            return f"churn/{key}"
        if re.match(r"graph_(?:rewire|utility|support|null|xsrc|xmass|coverage|changed)_l\d+$", key):
            return f"graph/{key.removeprefix('graph_')}"
        if key.startswith("codebook_"):
            return f"probe/{key}"
        return f"train/{key}"

    def write_entry(self, entry: dict) -> None:
        if entry["type"] == "churn_stats":
            self._pending_churn = {k: v for k, v in entry.items() if k != "type"}
            return
        if entry["type"] == "graph_stats":
            self._pending_graph = {k: v for k, v in entry.items() if k != "type"}
            return
        if entry["type"] == "train":
            self._last_train_time_ms = entry["train_time_ms"]
        elif entry["type"] == "val":
            if entry["train_time_ms"] == 0.0 and self._last_train_time_ms is not None:
                entry = {**entry, "train_time_ms": self._last_train_time_ms}
            if self._pending_churn:
                entry = {**entry, **self._pending_churn}
                self._pending_churn = None
            if self._pending_graph:
                entry = {**entry, **self._pending_graph}
                self._pending_graph = None
        self.metrics_file.write(json.dumps(entry, sort_keys=True) + "\n")
        self.metrics_file.flush()
        if entry["type"] == "val":
            time_s = int(entry["train_time_ms"] / 1000)
            self.writer.add_scalar("val/bpb", entry["val_bpb"], entry["step"])
            self.writer.add_scalar("val/loss", entry["val_loss"], entry["step"])
            self.writer.add_scalar("time/val_bpb", entry["val_bpb"], time_s)
            self.writer.add_scalar("time/val_loss", entry["val_loss"], time_s)
            for key, value in entry.items():
                if key in {"step", "type", "val_loss", "val_bpb", "train_time_ms"}:
                    continue
                self.writer.add_scalar(self._extra_scalar_tag(key), value, entry["step"])
        elif entry["type"] == "train":
            time_s = int(entry["train_time_ms"] / 1000)
            self.writer.add_scalar("train/loss", entry["train_loss"], entry["step"])
            self.writer.add_scalar("time/train_loss", entry["train_loss"], time_s)
            if self._prev_train_loss is not None:
                delta = entry["train_loss"] - self._prev_train_loss
                self.writer.add_scalar("train/loss_delta", delta, entry["step"])
            self._prev_train_loss = entry["train_loss"]
            for key, value in entry.items():
                if key in {"step", "type", "train_loss", "train_time_ms", "step_avg_ms"}:
                    continue
                self.writer.add_scalar(self._extra_scalar_tag(key), value, entry["step"])
            if "step_avg_ms" in entry:
                self.writer.add_scalar("perf/step_avg_ms", entry["step_avg_ms"], entry["step"])
        elif entry["type"] == "diag":
            for key, value in entry.items():
                if key in {"step", "type"}:
                    continue
                self.writer.add_scalar(self._extra_scalar_tag(key), value, entry["step"])

    def close(self) -> None:
        self.metrics_file.close()
        self.writer.close()


def read_metrics_jsonl(metrics_path: Path) -> list[dict]:
    entries = []
    if not metrics_path.exists():
        return entries
    with metrics_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def build_run_env(
    env_overrides: dict[str, str],
    steps: int,
    val_every: int,
    name: str,
) -> dict[str, str]:
    """Build a reproducible training environment for one ablation."""
    env = os.environ.copy()
    for key in list(env):
        if any(key.startswith(prefix) for prefix in SANITIZED_ENV_PREFIXES):
            env.pop(key)
    env.update(
        {
            "ITERATIONS": str(steps),
            "VAL_LOSS_EVERY": str(val_every),
            "TRAIN_LOG_EVERY": "10",
            "MAX_WALLCLOCK_SECONDS": "0",
            "PYTHONUNBUFFERED": "1",
            "RUN_ID": name,
        }
    )
    env.update(env_overrides)
    python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{python_path}" if python_path else str(REPO_ROOT)
    )
    if "WARMDOWN_ITERS" not in env_overrides:
        effective_steps = int(env["ITERATIONS"])
        default_warmdown_iters = (
            REFERENCE_WARMDOWN_ITERS
            if effective_steps >= REFERENCE_ABLATION_STEPS
            else 0
        )
        env["WARMDOWN_ITERS"] = str(default_warmdown_iters)
    return env


def run_config(
    name: str,
    env_overrides: dict[str, str],
    steps: int,
    val_every: int,
    script: str = "train_gpt.py",
) -> dict:
    """Run a single training config and return parsed results."""
    # Create result subfolder, clean stale data
    run_dir = RESULTS_DIR / name
    if run_dir.exists():
        import shutil
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_run_dir = TB_DIR / name
    if tb_run_dir.exists():
        import shutil
        shutil.rmtree(tb_run_dir)
    log_file = REPO_ROOT / "logs" / f"{name}.txt"
    if log_file.exists():
        log_file.unlink()
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    env = build_run_env(env_overrides, steps, val_every, name)
    effective_steps = int(env["ITERATIONS"])
    effective_val_every = int(env["VAL_LOSS_EVERY"])
    warmdown_iters_env = int(env["WARMDOWN_ITERS"])

    print(f"\n{'='*60}")
    print(f"  ABLATION: {name}")
    print(
        f"  steps={effective_steps}, val_every={effective_val_every}, "
        f"warmdown_iters_env={warmdown_iters_env}"
    )
    print(f"  script: {script}")
    if env_overrides:
        print(f"  overrides: {env_overrides}")
    print(f"{'='*60}\n")

    t0 = time.time()
    output_lines = []
    metrics_writer = MetricsWriter(metrics_path, name)
    proc = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", script],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            output_lines.append(line)
            entry = parse_log_line(line)
            if entry:
                metrics_writer.write_entry(entry)
        proc.wait()
    except BaseException:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise
    finally:
        metrics_writer.close()
    elapsed = time.time() - t0

    entries = read_metrics_jsonl(metrics_path)
    output_text = "".join(output_lines)

    val_entries = [e for e in entries if e["type"] == "val"]
    final_bpb = val_entries[-1]["val_bpb"] if val_entries else None
    final_loss = val_entries[-1]["val_loss"] if val_entries else None
    final_probe_bpb = val_entries[-1].get("probe_val_bpb") if val_entries else None
    final_probe_loss = val_entries[-1].get("probe_val_loss") if val_entries else None

    result = {
        "name": name,
        "script": script,
        "overrides": env_overrides,
        "steps": effective_steps,
        "val_every": effective_val_every,
        # This records the runner environment. Custom scripts may implement a
        # different schedule variable, so their own logs remain authoritative.
        "warmdown_iters_env": warmdown_iters_env,
        "elapsed_seconds": elapsed,
        "final_val_bpb": final_bpb,
        "final_val_loss": final_loss,
        "final_probe_val_bpb": final_probe_bpb,
        "final_probe_val_loss": final_probe_loss,
        "val_entries": val_entries,
        "returncode": proc.returncode,
    }

    if proc.returncode != 0:
        result["error"] = output_text[-2000:] if output_text else "unknown error"
        print(f"  ERROR (rc={proc.returncode})")
        print(output_text[-1000:])
    else:
        print(f"  Final BPB: {final_bpb:.4f}" if final_bpb else "  No val results found")
        print(f"  Elapsed: {elapsed:.1f}s")

    # Save result JSON; metrics.jsonl is the canonical machine-readable record.
    result_path = run_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {run_dir}/")

    return result


def compare_results(results_dir: Path) -> None:
    """Print a comparison table of all ablation results."""
    results = []
    # Support both old flat .json and new subfolder/result.json
    for f in sorted(results_dir.glob("*/result.json")):
        with open(f) as fh:
            results.append(json.load(fh))
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fh:
            results.append(json.load(fh))

    if not results:
        print("No results found.")
        return

    # Deduplicate by name
    seen = set()
    unique = []
    for r in results:
        if r["name"] not in seen:
            seen.add(r["name"])
            unique.append(r)

    print(
        f"\n{'Name':<40} {'Steps':>6} {'WD env':>7} "
        f"{'BPB':>8} {'Probe':>8} {'Loss':>8} {'Time':>8}"
    )
    print("-" * 91)
    for r in sorted(unique, key=lambda x: x.get("final_val_bpb") or 99):
        warmdown = r.get("warmdown_iters_env", r.get("warmdown_iters"))
        warmdown_text = str(warmdown) if warmdown is not None else "?"
        bpb = f"{r['final_val_bpb']:.4f}" if r.get("final_val_bpb") else "FAIL"
        probe = f"{r['final_probe_val_bpb']:.4f}" if r.get("final_probe_val_bpb") else "-"
        loss = f"{r['final_val_loss']:.4f}" if r.get("final_val_loss") else "-"
        time_s = f"{r['elapsed_seconds']:.0f}s" if r.get("elapsed_seconds") else "-"
        print(
            f"{r['name']:<40} {r['steps']:>6} {warmdown_text:>7} "
            f"{bpb:>8} {probe:>8} {loss:>8} {time_s:>8}"
        )


def build_sweep(sweep_type: str, steps: int, val_every: int) -> list[tuple[str, dict]]:
    """Build a list of (name, env_overrides) for a sweep."""
    if sweep_type == "lr":
        return [
            (f"lr_matrix_{lr}", {"MATRIX_LR": str(lr)})
            for lr in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
        ]
    elif sweep_type == "dim":
        return [
            (f"dim_{d}", {"MODEL_DIM": str(d), "NUM_HEADS": str(max(4, d // 64))})
            for d in [384, 512, 640, 768]
        ]
    elif sweep_type == "layers":
        return [
            (f"layers_{n}", {"NUM_LAYERS": str(n)})
            for n in [6, 9, 12, 15, 18]
        ]
    else:
        raise ValueError(f"Unknown sweep type: {sweep_type}. Use: lr, dim, layers")


def main():
    parser = argparse.ArgumentParser(description="Parameter Golf ablation runner")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps (default: 2000)")
    parser.add_argument("--val-every", type=int, default=None, help="Validate every N steps (default: 20)")
    parser.add_argument("--name", type=str, default=None, help="Run name")
    parser.add_argument("--compare", action="store_true", help="Compare existing results")
    parser.add_argument("--sweep", type=str, default=None, help="Run a preset sweep (lr, dim, layers)")
    parser.add_argument("--script", type=str, default="train_gpt.py", help="Training script to use")
    # action="extend" so repeated --env flags accumulate instead of the last
    # silently overwriting the rest (which drops e.g. DATA_PATH).
    parser.add_argument(
        "--env", type=str, nargs="*", action="extend", default=[],
        help="Extra env vars as KEY=VALUE (repeatable)",
    )
    args = parser.parse_args()

    if args.compare:
        compare_results(RESULTS_DIR)
        return

    val_every = args.val_every or 20
    extra_env = {}
    for kv in args.env:
        k, v = kv.split("=", 1)
        extra_env[k] = v

    if args.sweep:
        runs = build_sweep(args.sweep, args.steps, val_every)
        for name, overrides in runs:
            merged = {**extra_env, **overrides}
            run_config(name, merged, args.steps, val_every, args.script)
        print("\n\nSWEEP SUMMARY:")
        compare_results(RESULTS_DIR)
    else:
        name = args.name or f"baseline_s{args.steps}"
        run_config(name, extra_env, args.steps, val_every, args.script)


if __name__ == "__main__":
    main()
