"""Benchmark the production K3 LatentMoE at rollout and replay token counts.

Run through ``mlq``; this intentionally executes a CUDA model workload.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics

import torch

from pretraining.latent_moe import LatentMoEConfig, StableLatentMoE


def elapsed_ms(callable_, iterations: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        callable_()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return statistics.median(samples), p95


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="16,64,384,8192")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--implementations",
        default="grouped_mm,selected_bmm",
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument(
        "--quantile-balance",
        action="store_true",
        help="include compiled in-forward QB histogram accumulation",
    )
    parser.add_argument(
        "--correctness-reference",
        choices=("selected_bmm", "reference", "none"),
        default="selected_bmm",
    )
    args = parser.parse_args()
    if min(args.iterations, args.warmup) <= 0:
        parser.error("--iterations and --warmup must be positive")

    device = torch.device("cuda")
    config = LatentMoEConfig(
        model_dim=512,
        latent_dim=128,
        routed_hidden_dim=256,
        num_routed_experts=32,
        experts_per_token=2,
        shared_hidden_dim=64,
        num_shared_experts=2,
    )
    implementations = tuple(args.implementations.split(","))
    for tokens in map(int, args.tokens.split(",")):
        if tokens <= 0:
            parser.error("token counts must be positive")
        torch.manual_seed(17)
        module = StableLatentMoE(config, device=device).train(args.backward)
        if args.quantile_balance:
            module.enable_qb_collection(
                num_bins=1000, collect_in_eval=not args.backward
            )
        x = torch.randn(
            tokens,
            config.model_dim,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=args.backward,
        )
        with torch.no_grad():
            correctness_reference = (
                None
                if args.correctness_reference == "none"
                else module(x, implementation=args.correctness_reference).float()
            )
        gradient_baseline = None
        probe = torch.randn_like(x) if args.backward else None
        for implementation in implementations:
            def forward():
                return module(x, implementation=implementation)

            run_forward = torch.compile(forward, fullgraph=True) if args.compile else forward

            if args.backward:
                def run():
                    if args.quantile_balance:
                        module.reset_qb_accumulators_()
                    module.zero_grad(set_to_none=True)
                    x.grad = None
                    run_forward().float().square().mean().backward()
            else:
                @torch.no_grad()
                def run():
                    run_forward()

            with torch.no_grad():
                output = forward()
                max_abs_error = (
                    float("nan")
                    if correctness_reference is None
                    else float(
                        (output.float() - correctness_reference).abs().max()
                    )
                )
            gradient_max_abs_error = 0.0
            gradient_relative_l2_error = 0.0
            if args.backward:
                module.zero_grad(set_to_none=True)
                x.grad = None
                (forward().float() * probe.float()).sum().backward()
                gradients = {
                    name: parameter.grad.detach().clone()
                    for name, parameter in module.named_parameters()
                }
                gradients["__input__"] = x.grad.detach().clone()
                if gradient_baseline is None:
                    gradient_baseline = gradients
                else:
                    differences = [
                        (gradients[name].float() - reference.float()).reshape(-1)
                        for name, reference in gradient_baseline.items()
                    ]
                    references = [
                        reference.float().reshape(-1)
                        for reference in gradient_baseline.values()
                    ]
                    gradient_max_abs_error = float(
                        torch.stack([difference.abs().max() for difference in differences]).max()
                    )
                    gradient_relative_l2_error = float(
                        torch.cat(differences).norm()
                        / torch.cat(references).norm().clamp_min(1e-12)
                    )
            median_ms, p95_ms = elapsed_ms(run, args.iterations, args.warmup)
            print(
                json.dumps(
                    {
                        "tokens": tokens,
                        "implementation": implementation,
                        "compiled": args.compile,
                        "backward": args.backward,
                        "quantile_balance": args.quantile_balance,
                        "median_ms": median_ms,
                        "p95_ms": p95_ms,
                        "tokens_per_second": 1000.0 * tokens / median_ms,
                        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                        "max_abs_error_from_reference": max_abs_error,
                        "correctness_reference": args.correctness_reference,
                        "gradient_max_abs_error_from_first": gradient_max_abs_error,
                        "gradient_relative_l2_error_from_first": gradient_relative_l2_error,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del (
            run,
            run_forward,
            forward,
            module,
            x,
            probe,
            gradient_baseline,
            correctness_reference,
        )
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
