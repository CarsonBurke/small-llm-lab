"""Paired complete-update timing of dense event packing and query shifts.

Queue through mlq. Includes BF16 forward/backward, optimizer updates, all layout
work, and resident training/validation graphs. This is execution qualification,
not quality training; both arms start from the same completed checkpoint.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from scripts.benchmark_gated_delta import benchmark_candidate
from scripts.train_recurrent_slots import atomic_json
from pretraining.nanogpt_mini.gated_delta_dense_reads import QueryShiftSharedPoolGPT
from pretraining.nanogpt_mini.gated_delta_pool import SharedPoolGatedDeltaGPT
from pretraining.nanogpt_mini.gated_delta_runtime import (
    ForeignGpuSampler, wait_for_exclusive_gpu, runtime_dependency_versions,
    runtime_autotuning_policy, gdn2_dependency_provenance,
)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload["completed_steps"] != 1000:
        raise ValueError("Require the completed 1000-update qualification checkpoint")
    config = json.loads((args.checkpoint.parent / "config.json").read_text())
    if payload["model_config"] != config["model_config"]:
        raise ValueError("checkpoint configuration differs from canonical metadata")
    timing = Path(config["throughput_gate"]["path"])
    if digest(timing) != config["throughput_gate"]["sha256"]:
        raise ValueError("bound throughput report changed")
    evidence = json.loads(timing.read_text())
    sources = set(evidence["source_sha256"])
    sources.update(("pretraining/nanogpt_mini/gated_delta_dense_reads.py",
                    "scripts/benchmark_state_flow_execution.py",
                    "pretraining/tests/test_gated_delta_dense_reads_gpu.py"))
    for source, expected in evidence["source_sha256"].items():
        if digest(ROOT / source) != expected:
            raise ValueError(f"checkpoint implementation changed: {source}")
    if gdn2_dependency_provenance() != evidence["gdn2_dependency_provenance"]:
        raise ValueError("checkpoint kernel dependency changed")
    report = dict(status="running", checkpoint_sha256=digest(args.checkpoint),
                  batch_tokens=524288, seq_len=1024, microbatch=config["microbatch"],
                  dependency_versions=runtime_dependency_versions(),
                  autotuning_policy=runtime_autotuning_policy(),
                  gdn2_dependency_provenance=gdn2_dependency_provenance(),
                  source_sha256={p: digest(ROOT / p) for p in sorted(sources)})
    for source in sorted(sources):
        target = args.output / "sources" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / source).read_bytes())
    path = args.output / "benchmark.json"
    atomic_json(path, report)
    try:
        report["gpu_exclusivity"] = wait_for_exclusive_gpu()
        torch.manual_seed(config["seed"])
        inputs = torch.randint(payload["model_config"]["vocab_size"], (512, 1024),
                               device="cuda", dtype=torch.int32)
        targets = inputs.roll(-1, 1).long()
        for label, cls in (("events", SharedPoolGatedDeltaGPT), ("query_shift", QueryShiftSharedPoolGPT)):
            torch._dynamo.reset()
            torch.manual_seed(config["seed"])
            model = cls(**payload["model_config"])
            model.load_state_dict(payload["model"], strict=True)
            with ForeignGpuSampler() as sampler:
                report[label] = benchmark_candidate(model, inputs, targets,
                                                     microbatch=config["microbatch"], chunk_size=64, repeats=5)
            report[label]["gpu_exclusivity"] = sampler.summary()
            sampler.require_exclusive()
            print(json.dumps({label: report[label]}), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            atomic_json(path, report)
        old, new = report["events"], report["query_shift"]
        speedup = old["median_update_seconds"] / new["median_update_seconds"]
        separated = max(new["update_seconds"]) < min(old["update_seconds"])
        report.update(status="completed", speedup=speedup, repeated_timings_separated=separated,
                      execution_speed_gate_passed=speedup > 1.05 and separated)
        atomic_json(path, report)
        print(json.dumps({k: report[k] for k in ("status", "speedup", "execution_speed_gate_passed")}), flush=True)
        return 0 if report["execution_speed_gate_passed"] else 75
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
