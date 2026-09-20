"""Isolate packing effects on identical supervised rows; execute through mlq.

No optimizer updates. Only the first 64 rows contribute loss in both cases.
Compiled bf16 recurrence and full BPTT are retained. Hidden/logit diagnostics
resolve each token; slot snapshots resolve each 16-token checkpoint boundary.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.checkpoint import checkpoint

from pretraining.nanogpt_mini.recurrent_slots import RecurrentSlots
from pretraining.nanogpt_mini.recurrent_slots_runtime import RecurrentLoss
from scripts.train_recurrent_slots import atomic_json
from scripts.verify_recurrent_slots_packing import comparison


def temporal_comparison(actual, expected, stride=1):
    result = comparison(actual, expected)
    # Transfer only small scalar diagnostic arrays, never run a model on CPU.
    left, right = actual.double().flatten(2), expected.double().flatten(2)
    difference = left - right
    relative = difference.square().sum((0, 2)).sqrt() / right.square().sum((0, 2)).sqrt().clamp_min(1e-30)
    maximum = difference.abs().amax((0, 2))
    different = maximum > 0
    positions = different.nonzero().flatten()
    result.update(relative_error_by_position=relative.tolist(),
                  max_absolute_error_by_position=maximum.tolist(),
                  position_stride=stride,
                  first_different_position=(int(positions[0]) + 1) * stride if positions.numel() else None)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "ablation_results/recurrent_slots_packing_diagnostic")
    parser.add_argument("--lengths", nargs="+", type=int, default=[16, 128, 1024])
    parser.add_argument("--initial-memory", choices=["zero", "random"], default="zero")
    parser.add_argument("--strict-reductions", action="store_true",
                        help="disable bf16 reduced-precision matmul accumulation; retain default split-K")
    args = parser.parse_args(argv)
    if any(length < 16 or length > 1024 or length % 16 for length in args.lengths):
        parser.error("lengths must be multiples of16 between16 and1024")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Diagnostic requires CUDA bf16")
    if args.strict_reductions:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "diagnosis.json"
    report = {"status": "running", "optimizer_updates": 0, "supervised_rows": 64,
              "reference_rows": 64, "candidate_rows": 512, "initial_memory": args.initial_memory,
              "lengths": args.lengths, "segment_size": 16, "projection_std": 0.003,
              "memory_snapshot_stride": 16, "cases": [],
              "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__)}
    report.update(strict_reductions=args.strict_reductions,
                  allow_bf16_reduced_precision_reduction=
                  torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    atomic_json(path, report)
    begin = time.perf_counter()
    try:
        torch.manual_seed(1337)
        model = RecurrentSlots().cuda().train()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("proj.weight"):
                    parameter.normal_(std=0.003)
        loss_fn = RecurrentLoss(model)
        tokens = torch.randint(1024, (512, 1024), device="cuda", dtype=torch.int32)
        all_memory = model.initial_memory(512, tokens.device)
        if args.initial_memory == "random":
            all_memory.normal_(std=0.1)
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)

        def evaluate(rows, length):
            model.zero_grad(set_to_none=False)
            inputs = tokens[:rows, :length].contiguous()
            initial = all_memory[:rows].clone().requires_grad_()
            memory = initial
            states, memory_snapshots = [], []
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                for chunk in inputs.split(16, 1):
                    hidden, memory = checkpoint(model._compiled_segment, chunk, memory,
                                                 use_reentrant=False, preserve_rng_state=False)
                    states.append(hidden[:, :].contiguous())
                    memory_snapshots.append(memory[:64].detach().clone())
                hidden = torch.cat(states, 1)[:64]
                targets = tokens[:64, :length].roll(-1, 1).long().flatten()
                flat = hidden.reshape(-1, model.config["model_dim"])
                losses, logits = [], []
                for start in range(0, len(targets), 4096):
                    h, y = flat[start:start + 4096], targets[start:start + 4096]
                    losses.append(checkpoint(loss_fn.head_loss, h, y, use_reentrant=False,
                                             preserve_rng_state=False))
                    with torch.no_grad():
                        logits.append(model.logits(h).detach())
                loss = torch.stack(losses).sum()
                loss.backward()
            torch.cuda.synchronize()
            gradients = {}
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    raise AssertionError(f"Disconnected parameter: {name}")
                gradients[name] = parameter.grad.detach().float().clone()
            if initial.grad is None:
                raise AssertionError("Missing initial-memory gradient")
            unsupervised_gradient_max = float(initial.grad[64:].abs().max()) if rows > 64 else 0.0
            return dict(loss=float(loss.detach()), hidden=hidden.detach().clone(),
                        logits=torch.cat(logits).view(64, length, -1),
                        memory=torch.stack(memory_snapshots, 1),
                        initial_memory_gradient=initial.grad[:64].detach().clone(),
                        gradients=gradients, seconds=time.perf_counter() - started,
                        unsupervised_gradient_max=unsupervised_gradient_max)

        for length in args.lengths:
            print(f"diagnostic length={length} reference B64", flush=True)
            expected = evaluate(64, length)
            print(f"diagnostic length={length} candidate B512, first64 loss only", flush=True)
            actual = evaluate(512, length)
            case = {"length": length, "reference_loss": expected["loss"], "candidate_loss": actual["loss"],
                    "reference_seconds": expected["seconds"], "candidate_seconds": actual["seconds"],
                    "loss_relative_error": abs(actual["loss"] - expected["loss"]) / abs(expected["loss"]),
                    "hidden": temporal_comparison(actual["hidden"], expected["hidden"]),
                    "logits": temporal_comparison(actual["logits"], expected["logits"]),
                    "memory": temporal_comparison(actual["memory"], expected["memory"], 16),
                    "initial_memory_gradient": comparison(actual["initial_memory_gradient"], expected["initial_memory_gradient"]),
                    "unsupervised_initial_memory_gradient_max": actual["unsupervised_gradient_max"],
                    "gradients": {name: comparison(actual["gradients"][name], grad)
                                  for name, grad in expected["gradients"].items()}}
            report["cases"].append(case)
            atomic_json(path, report)
            print(f"length={length} hidden_relative={case['hidden']['relative_error']} "
                  f"initial_gradient_relative={case['initial_memory_gradient']['relative_error']}", flush=True)
            if actual["unsupervised_gradient_max"] != 0:
                raise AssertionError("Unsupervised rows received initial-memory gradients")
            del actual, expected
        report.update(status="completed", elapsed_seconds=time.perf_counter() - begin)
        atomic_json(path, report)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}",
                      elapsed_seconds=time.perf_counter() - begin)
        atomic_json(path, report)
        raise


if __name__ == "__main__":
    main()
