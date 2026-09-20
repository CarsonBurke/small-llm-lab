"""Complete-update Gated Delta versus plain-mini throughput; queue through mlq.

Five optimizer warmups and at least five timed updates per architecture measure
throughput only. No BPB claim or quality-training checkpoint is produced.
Both models receive identical 524288-token GPU-resident batches, summed CE,
compiled full losses, CUDA graph forward/backward, and AdamW/Muon updates.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path
import sys
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from scripts.benchmark_chunk_memory import PlainMini, benchmark
from scripts.train_recurrent_slots import atomic_json
from pretraining.nanogpt_mini.gated_delta_runtime import (
    CompiledGatedDeltaLoss, build_gated_delta_optimizers, runtime_dependency_versions,
)
from pretraining.nanogpt_mini.recurrent_slots_runtime import CUDAGraphMicrobatch, CUDAGraphValidation


def benchmark_candidate(model, inputs, targets, *, microbatch, chunk_size, repeats):
    model = model.cuda().train()
    loss_fn = CompiledGatedDeltaLoss(model, segment_size=chunk_size)
    opts = build_gated_delta_optimizers(model)
    model.eval()
    validation = CUDAGraphValidation(loss_fn, batch_size=microbatch, seq_len=1024)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        validation.replay(inputs[:microbatch], targets[:microbatch])
    model.train()
    graph = CUDAGraphMicrobatch(loss_fn, batch_size=microbatch, seq_len=1024)
    loss_fn.audit_graph_breaks()

    def update():
        graph.zero_grad()
        loss_sum = torch.zeros((), device="cuda")
        for row in range(0, 512, microbatch):
            loss_sum += graph.replay(inputs[row:row + microbatch], targets[row:row + microbatch])
        gradient_max = torch.stack(torch._foreach_norm(
            [p.grad for p in model.parameters()], float("inf"))).amax()
        if not bool(torch.isfinite(gradient_max) & torch.isfinite(loss_sum)):
            raise FloatingPointError("Nonfinite benchmark loss or gradient")
        for opt in opts:
            opt.step()
        return loss_sum

    # State initialization, optimizer compilation and first executions excluded.
    for _ in range(5):
        update()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples, losses = [], []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        loss = update()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
        losses.append(float(loss) / 524288)
    median = statistics.median(samples)
    result = dict(model_config=model.config, microbatch=microbatch,
                  chunk_size=chunk_size, slots=model.config.get("slots"),
                  parameters=sum(p.numel() for p in model.parameters()),
                  tokens_per_second=524288 / median, median_update_seconds=median,
                  update_seconds=samples, benchmark_losses=losses,
                  warmup_optimizer_updates=5, measured_optimizer_updates=repeats,
                  capture_seconds=graph.capture_seconds,
                  preparation_seconds=graph.preparation_seconds,
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    result["graph_breaks"] = loss_fn.audit_graph_breaks()
    result["execution"] = "compiled_dense_regions_official_triton_cuda_graph"
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        val_loss = validation.replay(inputs[:microbatch], targets[:microbatch])[0]
        if not bool(torch.isfinite(val_loss)):
            raise FloatingPointError("Nonfinite co-resident validation loss")
    result.update(validation_microbatch=microbatch, validation_residency_passed=True)
    model.train()
    del validation, graph, opts, loss_fn
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--head-dim", type=int, choices=(32, 64, 128), default=128)
    parser.add_argument("--mixer-dim", type=int, choices=(128, 256, 512), default=512)
    parser.add_argument("--fused-projections", action="store_true")
    parser.add_argument("--value-expansion", type=float, choices=(1.0, 2.0), default=1.0)
    parser.add_argument("--architecture", choices=("gdn1", "gdn2"), default="gdn2")
    parser.add_argument("--gdn-backend", choices=("vendor", "fla"), default="vendor")
    parser.add_argument("--state-v-first", action="store_true")
    parser.add_argument("--disable-recompute", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 5:
        parser.error("at least five complete timed updates are required")
    if args.architecture != "gdn2" and (args.gdn_backend != "vendor" or args.state_v_first or args.disable_recompute):
        parser.error("GDN2 execution options require --architecture gdn2")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Gated Delta throughput requires CUDA bf16")
    from pretraining.nanogpt_mini.gated_delta_model import GatedDeltaGPT
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "benchmark.json"
    sources = {ROOT / p for p in (
        "pretraining/nanogpt_mini/gated_delta_model.py",
        "pretraining/nanogpt_mini/gated_delta_runtime.py",
        "pretraining/nanogpt_mini/chunk_memory_runtime.py",
        "pretraining/nanogpt_mini/nanogpt_mini_model.py",
        "pretraining/nanogpt_mini/recurrent_slots_runtime.py",
        "scripts/benchmark_chunk_memory.py", "scripts/benchmark_gated_delta.py",
        "scripts/train_recurrent_slots.py")}
    if args.architecture == "gdn1":
        sources.add(ROOT / "pretraining/nanogpt_mini/scalar_delta_model.py")
    # Vendor paths are part of the gate, including provenance and license files.
    vendor = ROOT / "pretraining/gated_delta/vendor"
    if not vendor.is_dir():
        raise FileNotFoundError(f"Missing pinned kernel directory: {vendor}")
    sources.update(p for p in vendor.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    report = dict(status="running", gate_passed=False, batch_tokens=524288,
                  seq_len=1024, optimizer_included=True, repeats=args.repeats,
                  host_data_transfer_included=False, checkpointing=False,
                  compiled=True, cuda_graph=True,
                  candidate_execution="compiled_dense_regions_official_triton_cuda_graph",
                  optimizer="matched_mini_with_conv_adam_and_no_decay_gates",
                  precision="bf16_compute_embedding_fp32_other_parameters",
                  gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  dependency_versions=runtime_dependency_versions(),
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(sources)})
    if args.architecture == "gdn1":
        from pretraining.nanogpt_mini.gated_delta_runtime import scalar_dependency_provenance
        report["scalar_dependency_provenance"] = scalar_dependency_provenance()
    else:
        from pretraining.nanogpt_mini.gated_delta_runtime import gdn2_dependency_provenance
        report["gdn2_dependency_provenance"] = gdn2_dependency_provenance()
    # Keep an exact, inspectable snapshot even for failed capacity/speed probes.
    for source in sorted(sources):
        target = args.output / "sources" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    atomic_json(path, report)
    try:
        torch.manual_seed(1337)
        inputs = torch.randint(1024, (512, 1024), device="cuda", dtype=torch.int32)
        targets = inputs.roll(-1, 1).long()
        for label in ("baseline", "candidate"):
            torch.manual_seed(1337)
            if label == "baseline":
                model = PlainMini()
            elif args.architecture == "gdn1":
                from pretraining.nanogpt_mini.scalar_delta_model import ScalarDeltaGPT
                model = ScalarDeltaGPT(head_dim=args.head_dim, mixer_dim=args.mixer_dim,
                                       fused_projections=args.fused_projections, expand_v=args.value_expansion)
            else:
                model = GatedDeltaGPT(head_dim=args.head_dim, mixer_dim=args.mixer_dim,
                                      fused_projections=args.fused_projections, expand_v=args.value_expansion,
                                      gdn_backend=args.gdn_backend, state_v_first=args.state_v_first,
                                      disable_recompute=args.disable_recompute)
            print(f"benchmarking {label}: full-update compiled CUDA graphs", flush=True)
            runner = benchmark if label == "baseline" else benchmark_candidate
            extra = {"validation_microbatch": 64} if label == "baseline" else {}
            report[label] = runner(model, inputs, targets, microbatch=64 if label == "baseline" else args.microbatch,
                                      chunk_size=64, repeats=args.repeats, **extra)
            report[label]["model"] = ("nanogpt_mini" if label == "baseline" else
                                      "scalar_delta" if args.architecture == "gdn1" else "gated_delta")
            del model
            gc.collect()
            torch.cuda.empty_cache()
            atomic_json(path, report)
        baseline, candidate = report["baseline"], report["candidate"]
        speedup = candidate["tokens_per_second"] / baseline["tokens_per_second"]
        separated = max(candidate["update_seconds"]) < min(baseline["update_seconds"])
        report.update(status="completed", speedup=speedup,
                      median_gate_passed=speedup >= 1.05,
                      repeated_timings_separated=separated,
                      gate_passed=speedup >= 1.05 and separated)
        atomic_json(path, report)
        print(f"Gated Delta speedup={speedup:.4f}, gate_passed={report['gate_passed']}", flush=True)
        return 0 if report["gate_passed"] else 75
    except BaseException as error:
        report.update(status="failed", gate_passed=False, error=f"{type(error).__name__}: {error}")
        atomic_json(path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
