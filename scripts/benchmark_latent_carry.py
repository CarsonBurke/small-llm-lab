"""Full-update timing for latent-state reuse and refinement diagnostics.

Use mlq. Five warmup and five measured complete 524288-token optimizer updates
per arm, T1024, identical GPU-resident inputs. Refiner arms use B64/B128/B256;
plain-mini timing control is always B64. Different microbatch packing is
an explicit numerical variant, not a claim of identical optimizer trajectories.

A completed benchmark exits successfully even below the speed threshold, so
matched causal quality diagnostics can proceed. The 5% speed/separated-repeat
gate controls promotion separately from permission to measure quality.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from scripts.benchmark_chunk_memory import PlainMini
from scripts.train_latent_carry import (
    ARCHITECTURES, default_microbatch, validate_microbatch,
    make_model_and_loss, make_optimizers, source_hashes,
)
from scripts.train_recurrent_slots import atomic_json
from pretraining.nanogpt_mini.latent_carry_runtime import CausalControlLoss
from pretraining.nanogpt_mini.recurrent_slots_runtime import CUDAGraphMicrobatch, CUDAGraphValidation


def measure(model, loss_fn, inputs, targets, microbatch, repeats):
    model.train()
    optimizers = make_optimizers(model)
    before = dict(allocated_mib=torch.cuda.memory_allocated() / 2**20,
                  reserved_mib=torch.cuda.memory_reserved() / 2**20)
    torch.cuda.reset_peak_memory_stats()
    graph = CUDAGraphMicrobatch(loss_fn, batch_size=microbatch, seq_len=1024)
    capture_memory = dict(allocated_mib=torch.cuda.memory_allocated() / 2**20,
                          reserved_mib=torch.cuda.memory_reserved() / 2**20,
                          peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                          peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)

    def update():
        graph.zero_grad()
        total = torch.zeros((), device="cuda")
        for row in range(0, 512, microbatch):
            total += graph.replay(inputs[row:row + microbatch], targets[row:row + microbatch])
        gradients = [p.grad for p in model.parameters()]
        if any(g is None for g in gradients):
            raise RuntimeError("Disconnected benchmark parameter")
        maximum = torch.stack(torch._foreach_norm(gradients, float("inf"))).amax()
        if not bool(torch.isfinite(maximum) & torch.isfinite(total)):
            raise FloatingPointError("Nonfinite benchmark loss or gradient")
        for optimizer in optimizers:
            optimizer.step()
        graph.zero_grad()
        return total

    for _ in range(5):
        update()
    torch.cuda.synchronize()
    samples, losses = [], []
    for _ in range(repeats):
        started = time.perf_counter()
        loss = update()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
        losses.append(float(loss) / 524288)
    # Training retains both executors. Qualify that real memory requirement
    # after timing so validation preparation does not distort steady updates.
    training_peak = dict(allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                         reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    model.eval()
    validation = CUDAGraphValidation(loss_fn, batch_size=64, seq_len=1024)
    validation_loss = validation.replay(inputs[:64], targets[:64])[0]
    if not bool(torch.isfinite(validation_loss)):
        raise FloatingPointError("Nonfinite validation capacity check")
    torch.cuda.synchronize()
    model.train()
    graph.zero_grad()
    resumed_loss = graph.replay(inputs[:microbatch], targets[:microbatch])
    if not bool(torch.isfinite(resumed_loss)):
        raise FloatingPointError("Nonfinite training replay after validation capture")
    graph.zero_grad()
    torch.cuda.synchronize()
    return dict(model_config=model.config, microbatch=microbatch,
                parameters=sum(p.numel() for p in model.parameters()),
                tokens_per_second=524288 / statistics.median(samples),
                median_update_seconds=statistics.median(samples), update_seconds=samples,
                benchmark_losses=losses, warmup_optimizer_updates=5,
                training_only_peak=training_peak,
                training_and_validation_graphs_qualified=True,
                validation_microbatch=64,
                measured_optimizer_updates=repeats, memory_before_capture=before,
                memory_after_capture=capture_memory,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                capture_seconds=graph.capture_seconds, preparation_seconds=graph.preparation_seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--microbatch", type=int, choices=(64, 128, 256, 512))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.microbatch is None:
        args.microbatch = default_microbatch(args.architecture)
    if args.repeats < 5:
        parser.error("require at least five measured updates")
    try:
        validate_microbatch(args.architecture, args.microbatch)
    except ValueError as error:
        parser.error(str(error))
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Benchmark requires CUDA bf16")
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "benchmark.json"
    report = dict(status="running", architecture=args.architecture,
                  purpose="timing_for_matched_causal_diagnostic", quality_allowed_without_speed_gain=True,
                  gate_passed=False, batch_tokens=524288, seq_len=1024, optimizer_included=True,
                  compiled=True, cuda_graph=True, host_data_transfer_included=False,
                  same_execution_microbatch=args.microbatch == 64,
                  microbatch_numeric_variant=args.microbatch != 64, repeats=args.repeats,
                  source_sha256=source_hashes(), gpu=torch.cuda.get_device_name(),
                  torch=str(torch.__version__))
    atomic_json(path, report)
    try:
        torch.manual_seed(1337)
        inputs = torch.randint(1024, (512, 1024), device="cuda", dtype=torch.int32)
        targets = inputs.roll(-1, 1).long()
        for label in ("baseline", "candidate"):
            torch.manual_seed(1337)
            if label == "baseline":
                model = PlainMini().cuda()
                loss_fn = CausalControlLoss(model, segment_size=16)
                microbatch = 64
            else:
                model, loss_fn = make_model_and_loss(args.architecture)
                microbatch = args.microbatch
            print(f"benchmarking {label} {args.architecture} B{microbatch}: five warmups then {args.repeats} full updates", flush=True)
            report[label] = measure(model, loss_fn, inputs, targets, microbatch, args.repeats)
            report[label]["model"] = "nanogpt_mini" if label == "baseline" else args.architecture
            atomic_json(path, report)
            del model, loss_fn
            gc.collect()
            torch.cuda.empty_cache()
        candidate, baseline = report["candidate"], report["baseline"]
        ratio = candidate["tokens_per_second"] / baseline["tokens_per_second"]
        separated = max(candidate["update_seconds"]) < min(baseline["update_seconds"])
        report.update(status="completed", speedup=ratio, repeated_timings_separated=separated,
                      gate_passed=ratio >= 1.05 and separated)
        atomic_json(path, report)
        print(f"speedup={ratio:.5f}; promotion_speed_passed={report['gate_passed']}; quality_diagnostic_allowed=True", flush=True)
    except BaseException as error:
        report.update(status="failed", gate_passed=False, error=f"{type(error).__name__}: {error}")
        atomic_json(path, report)
        raise


if __name__ == "__main__":
    main()
