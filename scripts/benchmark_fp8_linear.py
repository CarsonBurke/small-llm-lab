from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from postraining.fast_inference import w8a16_linear


FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (
        weight.abs().amax(dim=1, keepdim=True).float() / FP8_MAX
    ).clamp_min(torch.finfo(torch.float32).tiny)
    quantized = (weight.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return quantized, scale


def build_bf16_linear():
    def bf16_linear(
        inputs: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        return F.linear(inputs, weight)

    return torch.compile(
        bf16_linear, fullgraph=True, mode="max-autotune"
    )


def build_weight_only_linear():
    return torch.compile(
        w8a16_linear,
        fullgraph=True,
        mode="max-autotune",
    )



def build_fp8_linear():
    def fp8_linear(
        inputs: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        input_scale = (
            inputs.abs().amax(dim=1, keepdim=True).float() / FP8_MAX
        ).clamp_min(torch.finfo(torch.float32).tiny)
        quantized_inputs = (
            (inputs.float() / input_scale)
            .clamp(-FP8_MAX, FP8_MAX)
            .to(FP8_DTYPE)
        )
        return torch.ops.aten._scaled_mm.default(
            quantized_inputs,
            weight.transpose(0, 1),
            input_scale,
            weight_scale.transpose(0, 1),
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

    return torch.compile(fp8_linear, fullgraph=True, mode="max-autotune")


def elapsed_microseconds(callable_, iterations: int) -> float:
    for _ in range(100):
        callable_()
    torch.cuda.synchronize()
    started = torch.cuda.Event(enable_timing=True)
    ended = torch.cuda.Event(enable_timing=True)
    started.record()
    for _ in range(iterations):
        callable_()
    ended.record()
    ended.synchronize()
    return started.elapsed_time(ended) * 1_000.0 / iterations


def benchmark_shape(
    rows: int,
    in_features: int,
    out_features: int,
    iterations: int,
) -> dict[str, float | int]:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(17_291)
    inputs = torch.randn(
        rows,
        in_features,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    weight = torch.randn(
        out_features,
        in_features,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ) / in_features**0.5
    quantized_weight, weight_scale = quantize_weight(weight)
    bf16_linear = build_bf16_linear()
    weight_only_linear = build_weight_only_linear()
    bf16_linear(inputs, weight)
    fp8_linear = build_fp8_linear()
    fp8_linear(inputs, quantized_weight, weight_scale)
    weight_only_linear(inputs, quantized_weight, weight_scale)
    torch.cuda.synchronize()

    torch.compiler.cudagraph_mark_step_begin()
    reference = bf16_linear(inputs, weight).clone()
    torch.compiler.cudagraph_mark_step_begin()
    candidate = fp8_linear(
        inputs, quantized_weight, weight_scale
    ).clone()
    torch.compiler.cudagraph_mark_step_begin()
    weight_only_candidate = weight_only_linear(
        inputs, quantized_weight, weight_scale
    ).clone()
    cosine = F.cosine_similarity(
        reference.float().flatten(), candidate.float().flatten(), dim=0
    )
    weight_only_cosine = F.cosine_similarity(
        reference.float().flatten(),
        weight_only_candidate.float().flatten(),
        dim=0,
    )
    bf16_us = elapsed_microseconds(
        lambda: bf16_linear(inputs, weight), iterations
    )
    fp8_us = elapsed_microseconds(
        lambda: fp8_linear(inputs, quantized_weight, weight_scale), iterations
    )
    weight_only_us = elapsed_microseconds(
        lambda: weight_only_linear(
            inputs, quantized_weight, weight_scale
        ),
        iterations,
    )
    return {
        "rows": rows,
        "in_features": in_features,
        "out_features": out_features,
        "bf16_microseconds": bf16_us,
        "fp8_microseconds": fp8_us,
        "speedup": bf16_us / fp8_us,
        "w8a16_microseconds": weight_only_us,
        "w8a16_speedup": bf16_us / weight_only_us,
        "w8a16_cosine_similarity": float(weight_only_cosine),
        "cosine_similarity": float(cosine),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Blackwell BF16, W8A8, and packed W8A16 linear layers."
    )
    parser.add_argument("--iterations", type=int, default=1_000)
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("iterations must be positive")

    shapes = (
        (1, 1_536, 1_536),
        (1, 1_536, 4_608),
        (1, 1_536, 130_560),
        (7, 1_536, 1_536),
        (7, 1_536, 4_608),
        (7, 1_536, 130_560),
        (16, 1_536, 1_536),
        (16, 1_536, 4_608),
        (16, 1_536, 130_560),
    )
    for shape in shapes:
        print(
            json.dumps(benchmark_shape(*shape, args.iterations), sort_keys=True),
            flush=True,
        )
        torch.compiler.reset()


if __name__ == "__main__":
    main()
