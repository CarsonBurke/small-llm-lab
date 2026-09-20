"""Verify 8x64 versus 1x512 bf16 gradient accumulation; submit through mlq.

Executes two complete 524288-token forward/backward evaluations with unchanged
parameters and no optimizer updates. It verifies numerical equivalence, not
bitwise equality or an empirical BPB improvement.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from pretraining.nanogpt_mini.recurrent_slots import RecurrentSlots
from pretraining.nanogpt_mini.recurrent_slots_runtime import RecurrentLoss
from scripts.train_recurrent_slots import atomic_json, muon_update


def comparison(actual, expected):
    """Accumulate diagnostic norms in fp64, including very small gradients."""
    lhs, rhs = actual.detach().double().flatten(), expected.detach().double().flatten()
    finite = bool(torch.isfinite(lhs).all() & torch.isfinite(rhs).all())
    if not finite:
        return {"finite": False, "relative_error": None, "cosine": None}
    left_norm, right_norm = lhs.norm(), rhs.norm()
    relative = float((lhs - rhs).norm() / right_norm.clamp_min(1e-30))
    if left_norm == 0 and right_norm == 0:
        cosine = 1.0
    elif left_norm == 0 or right_norm == 0:
        cosine = 0.0
    else:
        cosine = float(torch.dot(lhs, rhs) / (left_norm * right_norm))
    return {"finite": True, "reference_norm": float(right_norm),
            "actual_norm": float(left_norm), "relative_error": relative,
            "cosine": cosine, "max_absolute_error": float((lhs - rhs).abs().max())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "ablation_results/recurrent_slots_packing")
    parser.add_argument("--strict-reductions", action="store_true",
                        help="disable bf16 reduced-precision matmul accumulation; retain default split-K")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Packing verification requires CUDA bf16; no CPU fallback")
    if args.strict_reductions:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "optimizer_updates": 0,
              "batch_tokens": 524288, "rows": 512, "sequence_length": 1024,
              "reference_microbatch": 64, "candidate_microbatch": 512,
              "segment_size": 16, "seed": 1337, "projection_std": 0.003,
              "precision": "bf16_compute_embedding_fp32_other_parameters",
              "gradient_relative_tolerance": 0.05, "gradient_cosine_minimum": 0.995,
              "loss_relative_tolerance": 1e-5,
              "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__)}
    path = args.output / "verification.json"
    report.update(strict_reductions=args.strict_reductions,
                  allow_bf16_reduced_precision_reduction=
                  torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    atomic_json(path, report)
    started = time.perf_counter()
    try:
        torch.manual_seed(1337)
        model = RecurrentSlots().cuda().train()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("proj.weight"):
                    parameter.normal_(std=0.003)
        loss_fn = RecurrentLoss(model, segment_size=16)
        inputs = torch.randint(1024, (512, 1024), device="cuda", dtype=torch.int32)
        targets = inputs.roll(-1, 1).long()
        # Match training's stable accumulation buffers, including bf16 embed.
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)

        def evaluate(microbatch):
            model.zero_grad(set_to_none=False)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            total = torch.zeros((), device="cuda", dtype=torch.float64)
            for row in range(0, 512, microbatch):
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    loss = loss_fn(inputs[row:row + microbatch], targets[row:row + microbatch])
                    loss.backward()
                total += loss.detach().double()
            torch.cuda.synchronize()
            return float(total), time.perf_counter() - begin

        print("verifying reference: 8 x B64, full 1024-token BPTT", flush=True)
        reference_loss, reference_seconds = evaluate(64)
        reference_gradients = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                raise AssertionError(f"Disconnected reference parameter: {name}")
            reference_gradients[name] = parameter.grad.detach().float().clone()
        report.update(reference_loss=reference_loss, reference_seconds=reference_seconds)
        atomic_json(path, report)
        print("verifying candidate: 1 x B512, identical tokens and weights", flush=True)
        actual_loss, candidate_seconds = evaluate(512)
        errors, failures = {}, []
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                failures.append(f"Disconnected candidate parameter: {name}")
                continue
            item = comparison(parameter.grad, reference_gradients[name])
            errors[name] = item
            if (not item["finite"] or item["relative_error"] > 0.05
                    or item["cosine"] < 0.995):
                failures.append(f"Gradient packing disagreement: {name}: {item}")
        loss_relative_error = abs(actual_loss - reference_loss) / max(abs(reference_loss), 1e-30)
        if not math.isfinite(loss_relative_error) or loss_relative_error > 1e-5:
            failures.append(f"Summed loss relative error: {loss_relative_error}")
        report.update(candidate_loss=actual_loss, candidate_seconds=candidate_seconds,
                      loss_relative_error=loss_relative_error if math.isfinite(loss_relative_error) else None,
                      gradients=errors, failures=failures)
        atomic_json(path, report)
        print("checking all Muon matrix update directions with zero momentum", flush=True)
        muon_errors = {}
        special = {id(model.embed.weight), id(model.proj.weight), id(model.slot_identity)}
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.ndim < 2 or id(parameter) in special:
                    continue
                expected = reference_gradients[name]
                actual = parameter.grad.detach().float()
                if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
                    failures.append(f"Cannot compare nonfinite Muon input: {name}")
                    continue
                expected_update = muon_update(expected, torch.zeros_like(expected))
                actual_update = muon_update(actual, torch.zeros_like(actual))
                item = comparison(actual_update, expected_update)
                muon_errors[name] = item
                if (not item["finite"] or item["relative_error"] > 0.05
                        or item["cosine"] < 0.995):
                    failures.append(f"Muon direction packing disagreement: {name}: {item}")
        report.update(muon_update_directions=muon_errors, failures=failures,
                      status="failed" if failures else "passed",
                      elapsed_seconds=time.perf_counter() - started,
                      peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
        atomic_json(path, report)
        print(json.dumps({"status": report["status"], "report": str(path),
                          "loss_relative_error": report["loss_relative_error"],
                          "failures": failures}), flush=True)
        if failures:
            raise AssertionError("Packing verification failed; inspect verification.json")
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}",
                      elapsed_seconds=time.perf_counter() - started)
        atomic_json(path, report)
        raise


if __name__ == "__main__":
    main()
