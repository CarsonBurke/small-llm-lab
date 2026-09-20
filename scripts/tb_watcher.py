"""
Watch a metrics JSONL file and stream metrics to tensorboard.
Usage: python3 -m scripts.tb_watcher ablation_results/baseline_2k/metrics.jsonl --name baseline_2k
"""
import argparse
import json
import sys
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ablation import MetricsWriter, validation_primary

TB_DIR = Path(__file__).resolve().parents[1] / "tb_logs"


def parse_line(line):
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        return None
    if isinstance(entry, dict) and "type" in entry and "step" in entry:
        return entry
    return None


def write_entry(writer: SummaryWriter, entry: dict, prev_train_loss: float | None) -> float | None:
    if entry["type"] == "val":
        time_s = int(entry["train_time_ms"] / 1000)
        try:
            primary_key, primary = validation_primary(entry)
        except ValueError:
            return prev_train_loss
        primary_tag, time_tag = MetricsWriter.primary_scalar_tags(primary_key)
        writer.add_scalar(primary_tag, primary, entry["step"])
        writer.add_scalar("val/loss", entry["val_loss"], entry["step"])
        writer.add_scalar(time_tag, primary, time_s)
        writer.add_scalar("time/val_loss", entry["val_loss"], time_s)
        for key, value in entry.items():
            if key in {"step", "type", "val_loss", "val_bpb", "train_time_ms"}:
                continue
            writer.add_scalar(MetricsWriter._extra_scalar_tag(key), value, entry["step"])
        metric = (
            "document-stream-bpb"
            if primary_key == "val_bpb_document_stream"
            else "objective"
            if primary_key == "val_objective"
            else "diffusion-NELBO-bits/atom"
            if "val_diffusion_nelbo_bits_per_atom" in entry
            else "diffusion-elbo-proxy-bpb"
            if entry.get("val_diffusion_primary") == 1
            else "bpb"
        )
        print(
            f"  val step={entry['step']} {metric}={primary:.4f} "
            f"t={time_s}s"
        )
    elif entry["type"] == "train":
        time_s = int(entry["train_time_ms"] / 1000)
        writer.add_scalar("train/loss", entry["train_loss"], entry["step"])
        writer.add_scalar("time/train_loss", entry["train_loss"], time_s)
        if prev_train_loss is not None:
            writer.add_scalar("train/loss_delta", entry["train_loss"] - prev_train_loss, entry["step"])
        prev_train_loss = entry["train_loss"]
        for key, value in entry.items():
            if key in {"step", "type", "train_loss", "train_time_ms", "step_avg_ms"}:
                continue
            writer.add_scalar(MetricsWriter._extra_scalar_tag(key), value, entry["step"])
        if "step_avg_ms" in entry:
            writer.add_scalar("perf/step_avg_ms", entry["step_avg_ms"], entry["step"])
    elif entry["type"] == "diag":
        for key, value in entry.items():
            if key in {"step", "type"}:
                continue
            writer.add_scalar(MetricsWriter._extra_scalar_tag(key), value, entry["step"])
    return prev_train_loss


def read_new_lines(path: Path, position: int, partial_line: str) -> tuple[list[str], int, str]:
    try:
        size = path.stat().st_size
    except OSError:
        return [], position, partial_line

    if size < position:
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


def tail_and_write(metrics_path: str, name: str):
    writer = SummaryWriter(log_dir=str(TB_DIR / name), max_queue=512, flush_secs=10)
    position = 0
    partial_line = ""
    prev_train_loss = None
    print(f"Watching {metrics_path} -> tb_logs/{name}")

    try:
        while True:
            path = Path(metrics_path)
            lines, position, partial_line = read_new_lines(path, position, partial_line)
            for line in lines:
                e = parse_line(line)
                if not e:
                    continue
                prev_train_loss = write_entry(writer, e, prev_train_loss)
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_path", help="Path to metrics JSONL file")
    parser.add_argument("--name", required=True, help="Tensorboard run name")
    args = parser.parse_args()
    tail_and_write(args.metrics_path, args.name)
