"""Fair complete-update throughput gate; execute exclusively through mlq.

Both models use compiled, uncheckpointed whole-sequence losses and CUDA graphs.
Timing includes gradient zeroing, all eight microbatches, finite-gradient check,
and AdamW/Muon optimizer steps. Identical input data is GPU resident: host data
loading/transfer is excluded. Benchmark optimizer updates establish throughput,
not quality evidence, and produce no training checkpoint.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from pretraining.nanogpt_mini.nanogpt_mini_model import GPT
from pretraining.nanogpt_mini.chunk_memory_runtime import CompiledFullLoss
from pretraining.nanogpt_mini.recurrent_slots_runtime import CUDAGraphMicrobatch, CUDAGraphValidation
from scripts.train_recurrent_slots import Muon, atomic_json


class PlainMini(GPT):
    """Expose the unchanged plain mini backbone to the common compiled loss."""

    def __init__(self):
        super().__init__(1024, 6, 512)
        self.config = dict(vocab_size=1024, num_layers=6, model_dim=512)
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name.endswith("weight"):
                    if "proj" in name:
                        parameter.zero_()
                    elif "embed" in name:
                        parameter.normal_()
                    else:
                        parameter.normal_(std=math.sqrt(0.33 / parameter.shape[-1]))
                elif name.endswith("gains"):
                    parameter.fill_(1)
                else:
                    parameter.zero_()

    def forward_hidden(self, inputs, segment_size=256):
        hidden = self.norm1(self.embed(inputs))
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm2(hidden)
        return hidden, hidden[:, -1:, :]

    def logits(self, hidden):
        logits = self.proj(hidden).float()
        return 15 * logits * (logits.square() + 15**2).rsqrt()


def optimizers(model):
    special = {id(model.embed.weight), id(model.proj.weight)}
    groups = [{"params": [model.embed.weight], "lr": 0.7},
              {"params": [model.proj.weight], "lr": 0.004}]
    if hasattr(model, "slot_identity"):
        special.add(id(model.slot_identity))
        groups.append({"params": [model.slot_identity], "lr": 0.002, "weight_decay": 0.0})
    scalars = [p for p in model.parameters() if p.ndim < 2 and id(p) not in special]
    matrices = [p for p in model.parameters() if p.ndim >= 2 and id(p) not in special]
    groups.append({"params": scalars, "lr": 0.015})
    result = [torch.optim.AdamW(groups, betas=(0.8, 0.95), eps=1e-10,
                               weight_decay=0.001, fused=True), Muon(matrices)]
    grouped = [p for opt in result for group in opt.param_groups for p in group["params"]]
    if len(grouped) != len(set(grouped)) or set(grouped) != set(model.parameters()):
        raise RuntimeError("Optimizer groups must cover all parameters exactly once")
    return result


def benchmark(model, inputs, targets, *, microbatch, chunk_size, repeats, warmups=5,
              validation_microbatch=None):
    model = model.cuda().train()
    loss_fn = CompiledFullLoss(model, segment_size=chunk_size)
    opts = optimizers(model)
    validation = None
    if validation_microbatch is not None:
        model.eval()
        validation = CUDAGraphValidation(loss_fn, batch_size=validation_microbatch, seq_len=1024)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            validation.replay(inputs[:validation_microbatch], targets[:validation_microbatch])
        model.train()
    graph = CUDAGraphMicrobatch(loss_fn, batch_size=microbatch, seq_len=1024)

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
    for _ in range(warmups):
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
                  warmup_optimizer_updates=warmups, measured_optimizer_updates=repeats,
                  capture_seconds=graph.capture_seconds,
                  preparation_seconds=graph.preparation_seconds,
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    if validation is not None:
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            val_loss = validation.replay(inputs[:validation_microbatch], targets[:validation_microbatch])[0]
            if not bool(torch.isfinite(val_loss)):
                raise FloatingPointError("Nonfinite co-resident validation loss")
        result.update(validation_microbatch=validation_microbatch, validation_residency_passed=True)
        model.train()
    del validation, graph, opts, loss_fn
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, choices=(32, 64), default=64)
    parser.add_argument("--chunk-size", type=int, choices=(128, 256), default=256)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    if args.repeats < 5 or args.slots < 1:
        parser.error("require at least five timed updates and positive slots")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Throughput gate requires CUDA bf16")
    from pretraining.nanogpt_mini.chunk_memory import ChunkMemory
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "benchmark.json"
    sources = ["pretraining/nanogpt_mini/chunk_memory.py",
               "pretraining/nanogpt_mini/chunk_memory_runtime.py",
               "pretraining/nanogpt_mini/nanogpt_mini_model.py",
               "pretraining/nanogpt_mini/recurrent_slots_runtime.py",
               "scripts/benchmark_chunk_memory.py", "scripts/train_recurrent_slots.py"]
    report = dict(status="running", gate_passed=False, batch_tokens=524288,
                  seq_len=1024, host_data_transfer_included=False,
                  precision="bf16_compute_embedding_fp32_other_parameters",
                  checkpointing=False, compiled=True, cuda_graph=True,
                  optimizer_included=True, repeats=args.repeats,
                  gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources})
    atomic_json(path, report)
    try:
        torch.manual_seed(1337)
        inputs = torch.randint(1024, (512, 1024), device="cuda", dtype=torch.int32)
        targets = inputs.roll(-1, 1).long()
        for label in ("baseline", "candidate"):
            torch.manual_seed(1337)
            model = PlainMini() if label == "baseline" else ChunkMemory(slots=args.slots, chunk_size=args.chunk_size)
            print(f"benchmarking {label}: {args.repeats} full optimizer updates", flush=True)
            report[label] = benchmark(model, inputs, targets, microbatch=args.microbatch,
                                      chunk_size=args.chunk_size, repeats=args.repeats)
            report[label]["model"] = "nanogpt_mini" if label == "baseline" else "chunk_memory"
            del model
            gc.collect()
            torch.cuda.empty_cache()
            atomic_json(path, report)
        baseline, candidate = report["baseline"], report["candidate"]
        speedup = candidate["tokens_per_second"] / baseline["tokens_per_second"]
        separated = max(candidate["update_seconds"]) < min(baseline["update_seconds"])
        report.update(status="completed", speedup=speedup,
                      median_gate_passed=speedup >= 1.05, repeated_timings_separated=separated,
                      gate_passed=speedup >= 1.05 and separated)
        atomic_json(path, report)
        print(f"candidate speedup={speedup:.4f}, gate_passed={report['gate_passed']}", flush=True)
        return 0 if report["gate_passed"] else 75
    except BaseException as error:
        report.update(status="failed", gate_passed=False, error=f"{type(error).__name__}: {error}")
        atomic_json(path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
