"""Benchmark the emit readout tail: composed autograd vs the fused target op.

Times forward + backward of ``compact_emit_token_logprobs``'s vocabulary
tail at the KDA8 production readout shape (512 -> 50304, bf16 autocast),
both variants compiled exactly as the trainer compiles them
(``fullgraph=True, dynamic=True``), and reports each one's peak allocation
above the inputs. Queue through mlq:

    mlq submit --name bench_emit_readout --cwd "$PWD" --max-parallel-runs 1 \\
        -- .venv/bin/python -m postraining.bench_emit_readout
"""

from __future__ import annotations

import argparse
import json

import torch

from postraining.kda_backbone import NanoKDABackbone
from postraining.kda_gpu_parity import MODEL_KWARGS


def composed(backbone, features, targets):
    logits = backbone.logits_from_features(features)
    return logits.float().log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)


def fused(backbone, features, targets):
    return backbone.target_logprobs_from_features(features, targets)


def measure(function, backbone, features, targets, upstream, repeats):
    def step():
        backbone.zero_grad(set_to_none=True)
        features.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            value = function(backbone, features, targets)
        (value * upstream).sum().backward()
        return value

    for _ in range(3):
        value = step()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(repeats):
        step()
    stop.record()
    torch.cuda.synchronize()
    return {
        "ms": start.elapsed_time(stop) / repeats,
        "peak_above_inputs_gib": (torch.cuda.max_memory_allocated() - base) / 2**30,
    }, value.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--slots", type=int, nargs="+", default=[4096, 16384, 24576])
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    torch.manual_seed(0)
    # The production GPT-2 padded vocabulary; the trunk shape does not matter.
    backbone = NanoKDABackbone(**dict(MODEL_KWARGS, vocab_size=50304)).cuda().train()
    with torch.no_grad():
        backbone.proj.weight.normal_(std=0.2)
        backbone.proj.bias.normal_(std=1.0)
    compiled = {
        name: torch.compile(function, fullgraph=True, dynamic=True)
        for name, function in (("composed", composed), ("fused", fused))
    }
    results = []
    for slots in args.slots:
        features = torch.randn(
            slots, 2 * backbone.model_dim, device="cuda", requires_grad=True
        )
        targets = torch.randint(0, backbone.proj.out_features, (slots,), device="cuda")
        upstream = torch.randn(slots, device="cuda")
        row = {"slots": slots}
        values = {}
        for name, function in compiled.items():
            row[name], values[name] = measure(
                function, backbone, features, targets, upstream, args.repeats
            )
        row["max_abs_value_difference"] = float(
            (values["fused"] - values["composed"]).abs().max()
        )
        results.append(row)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
