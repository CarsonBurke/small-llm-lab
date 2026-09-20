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
import math
import os
import re
import signal
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
DECIMAL_PATTERN = r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
SCALAR_PATTERN = rf"(?:{DECIMAL_PATTERN}|[+-]?(?:nan|inf(?:inity)?))"
METRIC_INTEGRITY_RETURN_CODE = 65
TERMINATE_GRACE_SECONDS = 10.0


def parse_futility_gate(value: str) -> tuple[int, str, float]:
    """Parse ``STEP:METRIC:MAX`` for a validation-time early rejection gate."""

    step_text, separator, remainder = value.partition(":")
    metric, second_separator, threshold_text = remainder.partition(":")
    if not separator or not second_separator or not metric:
        raise ValueError(
            f"invalid futility gate {value!r}; expected STEP:METRIC:MAX"
        )
    step = int(step_text)
    threshold = float(threshold_text)
    if step <= 0 or not math.isfinite(threshold):
        raise ValueError("futility gate step must be positive and MAX finite")
    return step, metric, threshold


def terminate_process_group(
    proc: subprocess.Popen[str], *, grace_seconds: float | None = None
) -> dict[str, object]:
    """Bounded whole-tree termination for a rejected or interrupted run."""

    grace_seconds = TERMINATE_GRACE_SECONDS if grace_seconds is None else grace_seconds
    if grace_seconds < 0 or not math.isfinite(grace_seconds):
        raise ValueError("termination grace must be finite and nonnegative")
    process_group = proc.pid

    def group_exists() -> bool:
        try:
            os.killpg(process_group, 0)
            return True
        except ProcessLookupError:
            return False

    if proc.poll() is not None and not group_exists():
        return {"signal": None, "forced": False, "returncode": proc.returncode}
    if group_exists():
        os.killpg(process_group, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while group_exists() and time.monotonic() < deadline:
        proc.poll()
        time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
    if group_exists():
        os.killpg(process_group, signal.SIGKILL)
        proc.wait()
        return {
            "signal": signal.SIGKILL,
            "forced": True,
            "returncode": proc.returncode,
        }
    proc.wait()
    return {
        "signal": signal.SIGTERM,
        "forced": False,
        "returncode": proc.returncode,
    }


def _parse_extra_metrics(extras: str) -> tuple[dict[str, float], list[str]]:
    metrics: dict[str, float] = {}
    errors: list[str] = []
    for match in re.finditer(r"(?:^|\s)([a-zA-Z_][a-zA-Z0-9_]*):([^\s]+)", extras):
        key, token = match.groups()
        if key == "train_time" or key.startswith("native_"):
            continue
        if key == "step_avg" and token.endswith("ms"):
            token = token[:-2]
            # Compilers emit ``step_avg:nanms`` before enough timed steps
            # exist to compute an average. It means "unavailable", not that
            # the model produced a non-finite training metric.
            if token.lower() in {"nan", "+nan", "-nan"}:
                continue
        if re.fullmatch(SCALAR_PATTERN, token, flags=re.IGNORECASE) is None:
            errors.append(f"malformed metric {key}:{match.group(2)}")
            continue
        value = float(token)
        if not math.isfinite(value):
            errors.append(f"non-finite metric {key}:{match.group(2)}")
            continue
        if key == "step_avg":
            key = "step_avg_ms"
        metrics[key] = value
    return metrics, errors


def parse_extra_metrics(extras: str) -> dict[str, float]:
    metrics, errors = _parse_extra_metrics(extras)
    if errors:
        raise ValueError("; ".join(errors))
    return metrics


def parse_time_ms(text: str) -> float | None:
    match = re.search(
        rf"(?:^|\s)train_time:\s*({SCALAR_PATTERN})(ms|s|m)(?=\s|$)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = float(match.group(1))
    if not math.isfinite(value):
        return None
    unit = match.group(2).lower()
    if unit == "m":
        return value * 60_000
    if unit == "s":
        return value * 1_000
    return value


def _metric_error(line: str, message: str, *, step: int | None = None) -> dict:
    result: dict[str, object] = {
        "type": "metric_error",
        "metric_integrity_error": message,
        "raw_metric_line": line.rstrip("\r\n"),
    }
    if step is not None:
        result["step"] = step
    return result


def parse_log_line(line: str) -> dict | None:
    """Parse a single log line into a metric dict."""
    graph_match = re.match(r"graph_stats\s+(.*)", line)
    if graph_match:
        metrics, errors = _parse_extra_metrics(graph_match.group(1))
        if errors:
            return _metric_error(line, "; ".join(errors))
        return {"type": "graph_stats", **metrics} if metrics else None

    churn_match = re.match(r"churn_stats\s+(.*)", line)
    if churn_match:
        metrics, errors = _parse_extra_metrics(churn_match.group(1))
        if errors:
            return _metric_error(line, "; ".join(errors))
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
            metrics, errors = _parse_extra_metrics(extras)
            if errors:
                return _metric_error(
                    line, "; ".join(errors), step=int(diag_match.group(1))
                )
            entry.update(metrics)
        return entry

    val_match = re.match(
        rf"(?:step:)?(\d+)/\d+\s+val_loss:\s*({SCALAR_PATTERN})(?=\s|$)\s+(val_bpb|val_bpb_document_stream|val_diffusion_nelbo_bits_per_atom|val_objective):\s*({SCALAR_PATTERN})(?=\s|$)\s*(.*)",
        line,
        flags=re.IGNORECASE,
    )
    if val_match:
        step = int(val_match.group(1))
        val_loss = float(val_match.group(2))
        primary_key = val_match.group(3).lower()
        primary = float(val_match.group(4))
        if not math.isfinite(val_loss) or not math.isfinite(primary):
            return _metric_error(line, "non-finite validation loss or primary metric", step=step)
        extras = val_match.group(5).strip()
        train_time_ms = parse_time_ms(extras)
        if train_time_ms is None:
            if re.search(r"(?:^|\s)train_time:", extras):
                return _metric_error(line, "malformed train_time", step=step)
            train_time_ms = 0.0
        entry = {
            "step": step,
            "val_loss": val_loss,
            primary_key: primary,
            "train_time_ms": train_time_ms,
            "type": "val",
        }
        if extras:
            metrics, errors = _parse_extra_metrics(extras)
            if errors:
                return _metric_error(line, "; ".join(errors), step=step)
            entry.update(metrics)
        return entry
    train_match = re.match(
        rf"(?:step:)?(\d+)/\d+\s+train_loss:\s*({SCALAR_PATTERN})(?=\s|$)\s*(.*)",
        line,
        flags=re.IGNORECASE,
    )
    if train_match:
        step = int(train_match.group(1))
        train_loss = float(train_match.group(2))
        if not math.isfinite(train_loss):
            return _metric_error(line, "non-finite training loss", step=step)
        extras = train_match.group(3).strip()
        train_time_ms = parse_time_ms(extras)
        if train_time_ms is None:
            return _metric_error(line, "missing or malformed train_time", step=step)
        entry = {
            "step": step,
            "train_loss": train_loss,
            "train_time_ms": train_time_ms,
            "type": "train",
        }
        if extras:
            metrics, errors = _parse_extra_metrics(extras)
            if errors:
                return _metric_error(line, "; ".join(errors), step=step)
            entry.update(metrics)
        return entry
    malformed_metric = re.match(r"(?:step:)?(\d+)/\d+\s+(?:train_loss|val_loss):", line)
    if malformed_metric:
        return _metric_error(
            line,
            "malformed training or validation metric line",
            step=int(malformed_metric.group(1)),
        )
    return None


def validation_primary(entry: dict) -> tuple[str, float]:
    """Select an explicitly named metric without relabeling objectives as rates."""
    for key in ("val_bpb_document_stream", "val_bpb", "val_diffusion_nelbo_bits_per_atom", "val_objective"):
        if key in entry:
            value = entry[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"validation entry has non-finite {key}")
            return key, value
    raise ValueError("validation entry has no primary metric")


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
    def primary_scalar_tags(key: str) -> tuple[str, str]:
        # Keep BPB discoverable in the usual chart, without changing the
        # canonical metric's explicit evaluation scope or challenge eligibility.
        if key == "val_bpb_document_stream":
            return "val/bpb", "time/val_bpb"
        return MetricsWriter._extra_scalar_tag(key), f"time/{key}"

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
        if key.startswith("val_"):
            return f"val/{key.removeprefix('val_')}"
        if key == "step_avg_ms":
            return "perf/step_avg_ms"
        if key in {
            "train_wall_time_ms",
            "train_device_time_ms",
            "eval_seconds",
            "peak_vram_allocated_mib",
            "peak_vram_reserved_mib",
        }:
            return f"perf/{key}"
        if key.startswith("stage_") and key.endswith("_per_second"):
            return f"perf/{key}"
        if key == "train_ce":
            return "train/ce"
        if key.startswith("lejepa_"):
            return f"lejepa/{key.removeprefix('lejepa_')}"
        if key.startswith("nextlat_"):
            return f"nextlat/{key.removeprefix('nextlat_')}"
        if key.startswith("bolmo_"):
            return f"bolmo/{key.removeprefix('bolmo_')}"
        if key == "predicted_bytes_per_patch":
            return "bolmo/predicted_bytes_per_patch"
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
        self.metrics_file.write(
            json.dumps(entry, sort_keys=True, allow_nan=False) + "\n"
        )
        self.metrics_file.flush()
        if entry["type"] == "val":
            time_s = int(entry["train_time_ms"] / 1000)
            primary_key, primary = validation_primary(entry)
            primary_tag, time_tag = self.primary_scalar_tags(primary_key)
            self.writer.add_scalar(primary_tag, primary, entry["step"])
            self.writer.add_scalar("val/loss", entry["val_loss"], entry["step"])
            self.writer.add_scalar(time_tag, primary, time_s)
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


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _validate_finite_json(value: object, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON number at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json(item, f"{path}[{index}]")


def read_metrics_jsonl(metrics_path: Path) -> list[dict]:
    entries = []
    if not metrics_path.exists():
        return entries
    with metrics_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                )
                if not isinstance(entry, dict) or not isinstance(
                    entry.get("type"), str
                ):
                    raise ValueError("metric record must be an object with a type")
                _validate_finite_json(entry)
                entries.append(entry)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"malformed metrics JSON at {metrics_path}:{line_number}"
                ) from error
    return entries


def metric_integrity_errors(entries: list[dict], effective_steps: int) -> list[str]:
    errors = [
        str(entry["metric_integrity_error"])
        for entry in entries
        if entry.get("type") == "metric_error"
    ]
    final_validations = [
        entry
        for entry in entries
        if entry.get("type") == "val" and entry.get("step") == effective_steps
    ]
    for entry in final_validations:
        for key in ("val_loss", "train_time_ms"):
            value = entry.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                errors.append(
                    f"final validation has non-finite or missing {key}"
                )
        try:
            validation_primary(entry)
        except ValueError as error:
            errors.append(str(error))
    if not final_validations:
        errors.append(f"missing finite validation at requested step {effective_steps}")
    elif len(final_validations) != 1:
        errors.append(
            f"expected one validation at step {effective_steps}, "
            f"observed {len(final_validations)}"
        )
    return errors


def build_run_env(
    env_overrides: dict[str, str],
    steps: int,
    val_every: int,
    name: str,
) -> dict[str, str]:
    """Build a reproducible training environment for one ablation."""
    if "ABLATION_RUNNER_OWNS_METRICS" in env_overrides:
        raise ValueError("ABLATION_RUNNER_OWNS_METRICS is reserved by the runner")
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
            "ABLATION_RUNNER_OWNS_METRICS": "1",
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
    script_args: list[str] | None = None,
    futility_gates: tuple[tuple[int, str, float], ...] = (),
    stop_at: int | None = None,
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
    if any(step > effective_steps for step, _, _ in futility_gates):
        raise ValueError("futility gates cannot occur after the requested run")
    if len({step for step, _, _ in futility_gates}) != len(futility_gates):
        raise ValueError("futility gates must use unique validation steps")
    gates_by_step = {
        step: (metric, threshold) for step, metric, threshold in futility_gates
    }
    # --stop-at: the schedule (warmdown etc.) is that of the full run; the
    # process is terminated after the validation at this step and the result
    # is the metric at that step. Must land on a validation step.
    if stop_at is not None and (
        stop_at <= 0 or stop_at > effective_steps or stop_at % effective_val_every
    ):
        raise ValueError(
            f"--stop-at {stop_at} must be a validation step within the run "
            f"(val_every={effective_val_every}, steps={effective_steps})"
        )

    print(f"\n{'='*60}")
    print(f"  ABLATION: {name}")
    print(
        f"  steps={effective_steps}, val_every={effective_val_every}, "
        f"warmdown_iters_env={warmdown_iters_env}"
        + (f", stop_at={stop_at}" if stop_at is not None else "")
    )
    script_args = list(script_args or ())
    print(f"  script: {script}")
    if script_args:
        print(f"  script args: {script_args}")
    if env_overrides:
        print(f"  overrides: {env_overrides}")
    print(f"{'='*60}\n")

    t0 = time.time()
    output_lines = []
    metrics_writer = MetricsWriter(metrics_path, name)
    proc = None
    early_stop: dict[str, object] | None = None
    stopped_at: int | None = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", script, *script_args],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert proc.stdout is not None
        with log_file.open("w") as raw_log:
            for line in proc.stdout:
                output_lines.append(line)
                raw_log.write(line)
                raw_log.flush()
                entry = parse_log_line(line)
                if entry:
                    metrics_writer.write_entry(entry)
                    gate = gates_by_step.get(int(entry.get("step", -1)))
                    if entry.get("type") == "val" and gate is not None:
                        metric, threshold = gate
                        observed = entry.get(metric)
                        if (
                            isinstance(observed, bool)
                            or not isinstance(observed, (int, float))
                            or not math.isfinite(observed)
                        ):
                            raise RuntimeError(
                                f"futility gate metric {metric!r} is missing or non-finite"
                            )
                        if float(observed) >= threshold:
                            early_stop = {
                                "step": int(entry["step"]),
                                "metric": metric,
                                "observed": float(observed),
                                "reject_at_or_above": threshold,
                                "decision": "rejected_for_futility",
                            }
                            print(
                                "  FUTILITY STOP: "
                                f"step {entry['step']} {metric}={float(observed):.6f} "
                                f">= {threshold:.6f}",
                                flush=True,
                            )
                            early_stop["termination"] = terminate_process_group(
                                proc
                            )
                            break
                    if (
                        entry.get("type") == "val"
                        and stop_at is not None
                        and int(entry.get("step", -1)) == stop_at
                    ):
                        stopped_at = stop_at
                        print(f"  STOP AT: step {stop_at} (schedule of {effective_steps})", flush=True)
                        terminate_process_group(proc)
                        break
        proc.wait()
    except BaseException:
        if proc is not None and proc.poll() is None:
            terminate_process_group(proc)
        raise
    finally:
        metrics_writer.close()
    elapsed = time.time() - t0

    entries = read_metrics_jsonl(metrics_path)
    output_text = "".join(output_lines)

    val_entries = [e for e in entries if e["type"] == "val"]
    completed_steps = (
        int(early_stop["step"]) if early_stop is not None
        else stopped_at if stopped_at is not None
        else effective_steps
    )
    integrity_errors = metric_integrity_errors(entries, completed_steps)
    # A script that reports a codelength reports it under this key, and it is
    # the headline. Bolmo's `val_canonical_bpb` marginalizes a boundary bit
    # that its own routing consumed, so it is below any achievable codelength
    # and must not be compared against a subword model's bits-per-byte;
    # preferring it here is what put an unreachable number in every summary.
    # Subword runs emit no codelength key and fall through unchanged.
    final_entry = next(
        (
            entry
            for entry in reversed(val_entries)
            if entry["step"] == completed_steps
        ),
        None,
    )
    if integrity_errors:
        final_entry = None
    declared_generation_primary = Path(script).name in {
        "train_byte_idlm.py",
        "train_byte_diffusion_gemma.py",
        "train_byte_duo.py",
    }
    generation_primary = bool(
        declared_generation_primary
        or
        final_entry is not None
        and final_entry.get("generation_primary") == 1
    )
    final_bpb = None
    if final_entry is not None and not generation_primary:
        final_bpb = final_entry.get("val_canonical_codelength_bpb")
        if final_bpb is None:
            final_bpb = final_entry.get("val_challenge_bpb")
        if final_bpb is None and "val_proxy_bpb" not in final_entry:
            final_bpb = final_entry.get(
                "val_canonical_bpb", final_entry.get("val_bpb")
            )
    final_marginalized_bpb = (
        final_entry.get("val_canonical_bpb") if final_entry else None
    )
    final_loss = (
        final_entry.get("val_canonical_loss", final_entry["val_loss"])
        if final_entry
        else None
    )
    final_proxy_bpb = (
        final_entry.get("val_proxy_bpb", final_entry.get("val_bpb"))
        if final_entry and not generation_primary
        else None
    )
    final_ar_anchor_bpb = (
        final_entry.get("val_ar_anchor_bpb") if final_entry else None
    )
    final_proxy_loss = final_entry["val_loss"] if final_entry else None
    final_probe_bpb = final_entry.get("probe_val_bpb") if final_entry else None
    final_probe_loss = final_entry.get("probe_val_loss") if final_entry else None
    final_diffusion_elbo_proxy_bpb = (
        final_entry.get("val_diffusion_elbo_proxy_bpb") if final_entry else None
    )
    final_diffusion_nelbo_bits_per_atom = (
        final_entry.get("val_diffusion_nelbo_bits_per_atom")
        if final_entry
        else None
    )
    final_document_stream_bpb = (
        final_entry.get("val_bpb_document_stream") if final_entry else None
    )
    objective_primary = (
        final_entry is not None
        and validation_primary(final_entry)[0] == "val_objective"
    )
    promotion_metric = (
        "gsm8k_exact_match_generation_accuracy"
        if generation_primary
        else "document_stream_bpb"
        if final_document_stream_bpb is not None
        else "objective_only"
        if objective_primary
        else "diffusion_elbo_proxy_bpb"
        if final_diffusion_elbo_proxy_bpb is not None
        else "challenge_bpb" if final_bpb is not None else "proxy_bpb"
    )
    training_returncode = int(proc.returncode)
    expected_stop_returncodes = {0, -signal.SIGTERM, -signal.SIGKILL}
    expected_early_stop = (
        (early_stop is not None or stopped_at is not None)
        and training_returncode in expected_stop_returncodes
    )
    effective_returncode = 0 if expected_early_stop else training_returncode
    if effective_returncode == 0 and integrity_errors:
        effective_returncode = METRIC_INTEGRITY_RETURN_CODE

    result = {
        "name": name,
        "script": script,
        "script_args": script_args,
        "overrides": env_overrides,
        "steps": effective_steps,
        "completed_steps": completed_steps,
        "val_every": effective_val_every,
        # This records the runner environment. Custom scripts may implement a
        # different schedule variable, so their own logs remain authoritative.
        "warmdown_iters_env": warmdown_iters_env,
        "elapsed_seconds": elapsed,
        "final_val_bpb": final_bpb,
        "final_val_bpb_document_stream": final_document_stream_bpb,
        # Kept so the paper-comparable marginalized number stays recoverable
        # without re-reading metrics.jsonl. None for runs that never emit it.
        "final_val_marginalized_bpb": final_marginalized_bpb,
        "final_val_loss": final_loss,
        "final_proxy_val_bpb": final_proxy_bpb,
        "final_proxy_val_loss": final_proxy_loss,
        "final_ar_anchor_bpb": final_ar_anchor_bpb,
        "final_probe_val_bpb": final_probe_bpb,
        "final_probe_val_loss": final_probe_loss,
        "final_diffusion_elbo_proxy_bpb": final_diffusion_elbo_proxy_bpb,
        "final_diffusion_nelbo_bits_per_atom": (
            final_diffusion_nelbo_bits_per_atom
        ),
        "promotion_metric": promotion_metric,
        "val_entries": val_entries,
        "returncode": effective_returncode,
        "training_returncode": training_returncode,
        "metric_integrity_errors": integrity_errors,
        "futility_gates": [
            {"step": step, "metric": metric, "reject_at_or_above": threshold}
            for step, metric, threshold in futility_gates
        ],
        "early_stop": early_stop,
        "stop_at": stopped_at,
    }

    if early_stop is not None and expected_early_stop:
        print(
            "  Rejected by predeclared futility gate at "
            f"step {completed_steps}; training process rc={training_returncode}"
        )
    elif stopped_at is not None and expected_early_stop:
        metric = (
            "{}: {:.6g}".format(*validation_primary(final_entry))
            if final_entry is not None
            else "no validation result"
        )
        print(f"  Stopped at step {completed_steps} of {effective_steps}; {metric}")
    elif training_returncode != 0:
        result["error"] = output_text[-2000:] if output_text else "unknown error"
        print(f"  ERROR (rc={training_returncode})")
        print(output_text[-1000:])
    elif integrity_errors:
        result["error"] = "; ".join(integrity_errors)
        print(f"  METRIC INTEGRITY ERROR (rc={effective_returncode})")
        for error in integrity_errors:
            print(f"    - {error}")
    else:
        if final_bpb is not None:
            print(f"  Final BPB: {final_bpb:.4f}")
        elif final_document_stream_bpb is not None:
            print(f"  Final document-stream BPB (not packed challenge): {final_document_stream_bpb:.4f}")
        elif final_proxy_bpb is not None:
            print(f"  Final proxy BPB: {final_proxy_bpb:.4f}")
        elif objective_primary:
            print(f"  Final objective (not BPB): {final_loss:.6g}")
        else:
            print("  No val results found")
        print(f"  Elapsed: {elapsed:.1f}s")

    # Save result JSON; metrics.jsonl is the canonical machine-readable record.
    result_path = run_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(f"  Saved: {run_dir}/")

    return result


def compare_results(results_dir: Path) -> None:
    """Print a comparison table of all ablation results."""
    results = []
    # Support both old flat .json and new subfolder/result.json
    for f in sorted(results_dir.glob("*/result.json")):
        with open(f) as fh:
            result = json.load(fh, parse_constant=_reject_json_constant)
            _validate_finite_json(result)
            results.append(result)
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fh:
            result = json.load(fh, parse_constant=_reject_json_constant)
            _validate_finite_json(result)
            results.append(result)

    if not results:
        print("No results found.")
        return

    # Deduplicate by name
    seen = set()
    unique = []
    for r in results:
        if not isinstance(r, dict) or "name" not in r:
            continue
        if r["name"] not in seen:
            seen.add(r["name"])
            unique.append(r)

    print(
        f"\n{'Name':<40} {'Steps':>9} {'WD env':>7} "
        f"{'Scope':>9} {'BPB':>8} {'Probe':>8} {'Loss':>8} {'Time':>8}"
    )
    print("-" * 101)

    def comparison_bpb(result: dict) -> float | None:
        if result.get("returncode", 0) != 0 or result.get("early_stop") is not None:
            return None
        if result.get("promotion_metric") in {"gsm8k_exact_match_generation_accuracy", "objective_only"}:
            return None
        return (
            result.get("final_val_bpb_document_stream")
            or result.get("final_val_bpb")
            or result.get("final_proxy_val_bpb")
        )

    def comparison_order(result: dict) -> tuple[int, float]:
        value = comparison_bpb(result)
        if value is not None:
            return (0, value)
        if (
            result.get("early_stop") is not None
        ):
            return (2, 0.0)
        if (
            result.get("returncode", 0) == 0
            and result.get("promotion_metric")
            == "gsm8k_exact_match_generation_accuracy"
        ):
            return (1, 0.0)
        return (3, 0.0)

    for r in sorted(unique, key=comparison_order):
        warmdown = r.get("warmdown_iters_env", r.get("warmdown_iters"))
        warmdown_text = str(warmdown) if warmdown is not None else "?"
        value = comparison_bpb(r)
        generation_primary = (
            r.get("returncode", 0) == 0
            and r.get("early_stop") is None
            and r.get("promotion_metric")
            == "gsm8k_exact_match_generation_accuracy"
        )
        objective_primary = (
            r.get("returncode", 0) == 0
            and r.get("promotion_metric") == "objective_only"
        )
        bpb = (
            f"{value:.4f}"
            if value is not None
            else "REJECT"
            if r.get("early_stop") is not None
            else "GEN"
            if generation_primary
            else "OBJ"
            if objective_primary
            else "FAIL"
        )
        scope = "challenge" if r.get("final_val_bpb") is not None else "proxy"
        if r.get("final_val_bpb_document_stream") is not None:
            scope = "docstream"
        if r.get("early_stop") is not None:
            scope = "futility"
        elif generation_primary:
            scope = "gsm8k"
        elif objective_primary:
            scope = "objective"
        elif value is None:
            scope = "-"
        probe = f"{r['final_probe_val_bpb']:.4f}" if r.get("final_probe_val_bpb") else "-"
        loss = f"{r['final_val_loss']:.4f}" if r.get("final_val_loss") else "-"
        time_s = f"{r['elapsed_seconds']:.0f}s" if r.get("elapsed_seconds") else "-"
        steps_text = (
            f"{r['completed_steps']}/{r['steps']}"
            if r.get("stop_at") is not None else str(r["steps"])
        )
        print(
            f"{r['name']:<40} {steps_text:>9} {warmdown_text:>7} "
            f"{scope:>9} {bpb:>8} {probe:>8} {loss:>8} {time_s:>8}"
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Parameter Golf ablation runner")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps (default: 2000)")
    parser.add_argument("--val-every", type=int, default=None, help="Validate every N steps (default: 20)")
    parser.add_argument("--name", type=str, default=None, help="Run name")
    parser.add_argument("--compare", action="store_true", help="Compare existing results")
    parser.add_argument("--sweep", type=str, default=None, help="Run a preset sweep (lr, dim, layers)")
    parser.add_argument("--script", type=str, default="train_gpt.py", help="Training script to use")
    parser.add_argument(
        "--script-arg",
        action="append",
        default=[],
        help=(
            "Argument forwarded verbatim to the training script; repeat once per "
            "argument (use --script-arg=--flag for values beginning with a dash)"
        ),
    )
    parser.add_argument(
        "--futility-gate",
        action="append",
        default=[],
        metavar="STEP:METRIC:MAX",
        help=(
            "terminate and record a rejected run when a validation metric is "
            "greater than or equal to MAX at STEP; repeat for multiple gates"
        ),
    )
    parser.add_argument(
        "--stop-at", type=int, default=None, metavar="STEP",
        help=(
            "terminate after the validation at STEP while keeping the full "
            "--steps schedule; the result is the metric at STEP"
        ),
    )
    # action="extend" so repeated --env flags accumulate instead of the last
    # silently overwriting the rest (which drops e.g. DATA_PATH).
    parser.add_argument(
        "--env", type=str, nargs="*", action="extend", default=[],
        help="Extra env vars as KEY=VALUE (repeatable)",
    )
    args = parser.parse_args()
    futility_gates = tuple(map(parse_futility_gate, args.futility_gate))

    if args.compare:
        compare_results(RESULTS_DIR)
        return 0

    val_every = args.val_every or 20
    extra_env = {}
    for kv in args.env:
        k, v = kv.split("=", 1)
        extra_env[k] = v

    if args.sweep:
        runs = build_sweep(args.sweep, args.steps, val_every)
        returncode = 0
        for name, overrides in runs:
            merged = {**extra_env, **overrides}
            result = run_config(
                name,
                merged,
                args.steps,
                val_every,
                args.script,
                args.script_arg,
                futility_gates,
                args.stop_at,
            )
            returncode = returncode or int(result["returncode"])
        print("\n\nSWEEP SUMMARY:")
        compare_results(RESULTS_DIR)
        return returncode
    else:
        name = args.name or f"baseline_s{args.steps}"
        result = run_config(
            name,
            extra_env,
            args.steps,
            val_every,
            args.script,
            args.script_arg,
            futility_gates,
            args.stop_at,
        )
        return int(result["returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
