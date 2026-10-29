#!/usr/bin/env python3
"""Queued CUDA tensor audit of original vs stable compiled token readouts.

No model, checkpoint, rollout or optimizer is executed. Inputs reproduce the
production bf16 dtype, vocabulary width and nearly deterministic softmax.
"""

import argparse
import ast
import json
import sys
import types
from pathlib import Path

import torch
from torch import Tensor

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-source", type=Path, required=True)
    parser.add_argument("--new-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("queued CUDA required; no CPU fallback")
    def readout(path):
        names = {"rational_softcap", "_shifted_softcapped_logits", "SoftcappedTargetLogprobs"}
        tree = ast.parse(path.read_text())
        subset = ast.Module(body=[node for node in tree.body if getattr(node, "name", None) in names], type_ignores=[])
        module = types.ModuleType(path.stem)
        sys.modules[path.stem] = module
        namespace = module.__dict__
        namespace.update(torch=torch, Tensor=Tensor)
        exec(compile(subset, str(path), "exec"), namespace)
        return torch.compile(namespace["SoftcappedTargetLogprobs"].apply, fullgraph=True, dynamic=True)
    old = readout(args.old_source)
    new = readout(args.new_source)
    results = []
    for rows in (1, 8, 128, 512):
        targets = torch.arange(rows, device="cuda") % 50304
        raw = torch.full((rows, 50304), -90., device="cuda", dtype=torch.bfloat16)
        chosen = torch.linspace(10., 512., rows, device="cuda").to(raw.dtype)
        raw.scatter_(1, targets[:, None], chosen[:, None])
        raw.requires_grad_()
        for cap in (15., 30.):
            baseline = old(raw, targets, cap)
            revised = new(raw, targets, cap)
            x = raw.double()
            capped = cap * x * (x.square() + cap**2).rsqrt()
            reference = capped.log_softmax(-1).gather(1, targets[:, None]).squeeze(1)
            assert torch.isfinite(revised).all(), "stable readout produced nonfinite values"
            assert (revised <= 0).all(), "stable readout produced positive log probabilities"
            for label, values in (("original", baseline), ("stable", revised)):
                gradient, = torch.autograd.grad(values.sum(), raw)
                assert torch.isfinite(gradient).all(), f"{label} readout produced nonfinite gradients"
                results.append({"rows": rows, "cap": cap, "implementation": label,
                                "positive_count": int((values > 0).sum()),
                                "nonfinite_count": int((~torch.isfinite(values)).sum()),
                                "maximum_logprob": float(values.max()),
                                "max_abs_gradient": float(gradient.abs().max()),
                                "max_abs_reference_error": float((values.double() - reference).abs().max())})
        del raw, capped, x
    args.output.write_text(json.dumps({"scope": "compiled CUDA bf16 tensor readouts; synthetic peaked policies, no model", "results": results}, indent=2) + "\n")
    print(json.dumps({"passed": True, "original_positive_cases": sum(r["positive_count"] for r in results if r["implementation"] == "original"), "stable_positive_cases": sum(r["positive_count"] for r in results if r["implementation"] == "stable"), "output": str(args.output)}))


if __name__ == "__main__":
    main()
