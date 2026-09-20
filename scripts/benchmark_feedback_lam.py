#!/usr/bin/env python3
"""Benchmark compiled production-shape LAM; submit the entire command through mlq.

Synthetic forward/backward timings are not BPB evidence or optimizer-update timings.
Numerical contracts live in pretraining/tests/test_nanogpt_mini_feedback_gpu.py.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
CONFIGURATIONS = ("none", "lam_torch", "lam_fused_legacy", "lam_fused_compiled",
                  "lam_first", "lam_alternate", "glu")
GEOMETRY = dict(microbatch=64, sequence_length=1024, vocab_size=1024, num_layers=6, model_dim=512)


def persist(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def stats(samples: list[float]) -> dict:
    return dict(samples=samples, median=statistics.median(samples),
                mean=statistics.mean(samples), stdev=statistics.pstdev(samples),
                min=min(samples), max=max(samples))


def worker(args) -> dict:
    os.environ["FB_MEMORY_KERNEL"] = "0" if args.worker == "lam_torch" else "1"
    os.environ["FB_MEMORY_COMPILED"] = "0" if args.worker == "lam_fused_legacy" else "1"
    sys.path.insert(0, str(REPO))
    import torch
    from pretraining.nanogpt_mini.nanogpt_mini_feedback_model import FeedbackGPT

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with bf16 support is required; no CPU/fp32 fallback")
    torch.cuda.set_device(0)
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    mode = "none" if args.worker == "none" else "glu" if args.worker == "glu" else "lam"
    memory_layers = {"lam_first": (0,), "lam_alternate": (0, 2, 4)}.get(args.worker)
    model = FeedbackGPT(1024, 6, 512, mode=mode, noise=0.02,
                        memory_layers=memory_layers).cuda().train()
    # Match trainer initialization and RNG consumption, not just distributions.
    torch.manual_seed(args.seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("weight"):
                if "proj" in name:
                    parameter.zero_()
                elif "embed" in name:
                    parameter.normal_()
                else:
                    parameter.normal_(std=0.33**0.5 / parameter.size(-1)**0.5)
            elif name.endswith("bias"):
                parameter.zero_()
            elif name.endswith("gains"):
                parameter.normal_(mean=1, std=0)
            elif name not in ("decay_logit", "temperature"):
                raise ValueError(f"Uninitialized parameter: {name}")
        model.reset_extra_parameters()
    generator = torch.Generator().manual_seed(args.seed + 1)
    tokens = torch.randint(1024, (64 * 1024 + 1,), generator=generator, dtype=torch.int32)
    inputs = tokens[:-1].reshape(64, 1024).cuda()
    targets = tokens[1:].long().reshape(64, 1024).cuda()
    compile_settings = dict(dynamic=False, fullgraph=args.worker != "lam_fused_legacy", mode="default")
    model.compile(**compile_settings)
    passes = 1 if args.worker == "none" else 2
    events = [tuple(torch.cuda.Event(enable_timing=True) for _ in range(3))
              for _ in range(args.accumulation)]

    def repetition(measured: bool) -> dict:
        loss = None
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for before, middle, after in events:
            if measured:
                before.record()
            loss = model(inputs, targets, passes)
            if measured:
                middle.record()
            loss.backward()
            if measured:
                after.record()
        torch.cuda.synchronize()
        elapsed = 1000 * (time.perf_counter() - start)
        assert loss is not None
        if not torch.isfinite(loss).item():
            raise RuntimeError("Nonfinite benchmark loss")
        if not measured:
            return {}
        return dict(wall_ms=elapsed,
                    forward_gpu_ms=sum(a.elapsed_time(b) for a, b, _ in events),
                    backward_gpu_ms=sum(b.elapsed_time(c) for _, b, c in events))

    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.warmup):
        repetition(False)
    warmup_peak = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    measured = [repetition(True) for _ in range(args.samples)]
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError(f"Missing/nonfinite gradient: {name}")
    timings = {key: stats([sample[key] for sample in measured]) for key in measured[0]}
    microbatch_ms = timings["wall_ms"]["median"] / args.accumulation
    return dict(status="passed", configuration=args.worker, geometry=GEOMETRY,
                parameters=sum(p.numel() for p in model.parameters()), compile=compile_settings,
                initialization="trainer_zero_projection", accumulation=args.accumulation,
                memory_layers=model.config.get("memory_layers"), mode=mode,
                warmup=args.warmup, samples=args.samples, seed=args.seed, noise=0.02,
                environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                 gpu=torch.cuda.get_device_name(), python=sys.version,
                                 matmul_precision=torch.get_float32_matmul_precision()),
                timings=timings, median_microbatch_wall_ms=microbatch_ms,
                median_tokens_per_second=65536 * 1000 / microbatch_ms,
                measured_peak_memory=dict(allocated_bytes=allocated, reserved_bytes=reserved),
                warmup_peak_allocated_bytes=warmup_peak)


def driver(args) -> int:
    result: dict = dict(status="running", configurations={}, geometry=GEOMETRY,
                  limitations=["Synthetic repeated inputs; no BPB or training-quality claim.",
                               "Excludes optimizer, communication, data loading and gradient clearing.",
                               "Only the legacy FLA arm permits its known graph breaks.",
                               "CUDA event timings include stream idle time; reserved VRAM includes warmup."])
    persist(args.output, result)
    with tempfile.TemporaryDirectory(prefix="feedback-lam-") as directory:
        for name in args.configurations:
            output = Path(directory) / f"{name}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", name,
                       "--output", str(output), "--samples", str(args.samples),
                       "--warmup", str(args.warmup), "--accumulation", str(args.accumulation),
                       "--seed", str(args.seed)]
            print(f"Running {name}", flush=True)
            # Foreground children share mlq's process group; exit releases their CUDA state.
            child = subprocess.run(command, cwd=REPO, check=False)
            entry: dict = json.loads(output.read_text()) if output.exists() else dict(status="failed", error="No worker result")
            entry["exit_code"] = child.returncode
            if child.returncode:
                entry["status"] = "failed"
            result["configurations"][name] = entry
            persist(args.output, result)
    passed = all(entry["status"] == "passed" for entry in result["configurations"].values())
    result["status"] = "passed" if passed else "failed"
    persist(args.output, result)
    print(json.dumps({name: {key: entry.get(key) for key in ("status", "median_microbatch_wall_ms", "measured_peak_memory")}
                      for name, entry in result["configurations"].items()}), flush=True)
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configurations", nargs="+", choices=CONFIGURATIONS, default=list(CONFIGURATIONS))
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--accumulation", type=int, choices=(1, 8), default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--worker", choices=CONFIGURATIONS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples < 2 or args.warmup < 1:
        parser.error("Require at least two measured samples and one warmup")
    if args.worker is None:
        return driver(args)
    try:
        result = worker(args)
    except Exception as error:
        result = dict(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    persist(args.output, result)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
