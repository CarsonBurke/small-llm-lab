"""Run MiniCPM VAPO with the existing carry-run JSONL phase/metrics observer."""
from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import postraining.train_minicpm_vapo as training


def main() -> None:
    args = training.build_parser().parse_args()
    training._validate_args(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    phase = "initializing"
    with (output / "metrics.jsonl").open("a", buffering=1) as stream:
        def emit(event, **values):
            row = {
                "event": event, "phase": phase, "wall_time": time.time(),
                "elapsed_seconds": time.monotonic() - started, **values,
            }
            line = json.dumps(row, allow_nan=False)
            stream.write(line + "\n")
            print(line, flush=True)

        original_scalars = training.tensorboard_scalars

        @functools.wraps(original_scalars)
        def scalars(writer, namespace, metrics, step):
            original_scalars(writer, namespace, metrics, step)
            if namespace != "config" and not namespace.endswith("_live"):
                emit("metrics", namespace=namespace, step=step, metrics=metrics)
                writer.flush()

        training.tensorboard_scalars = scalars

        def observe(name):
            original = getattr(training, name)

            @functools.wraps(original)
            def run(*call_args, **kwargs):
                nonlocal phase
                previous = phase
                phase = name
                emit("phase_start")
                begin = time.monotonic()
                result = original(*call_args, **kwargs)
                torch.cuda.synchronize()
                emit("phase_end", seconds=time.monotonic() - begin)
                phase = previous
                return result

            setattr(training, name, run)

        for name in (
            "collect_rollouts", "update_step", "refresh_behavior_statistics", "save_checkpoint",
        ):
            observe(name)
        emit(
            "start", args=sys.argv[1:],
            source_sha256={
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in (
                    "postraining/token_carry.py", "postraining/slot_memory.py",
                    "postraining/vapo/policy.py",
                    "postraining/train_minicpm_vapo.py", "postraining/fast_inference.py",
                    "postraining/core.py", "scripts/run_minicpm_vapo.py",
                )
            },
        )
        try:
            training.main()
            emit("completed")
        except BaseException as error:
            emit("failed", error=repr(error))
            raise


if __name__ == "__main__":
    main()
