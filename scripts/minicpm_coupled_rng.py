#!/usr/bin/env python3
"""Bounded CUDA RNG qualification; queue through mlq with --queued-run."""

import argparse
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from postraining.runtime import coupled_rng


def _qualify():
    """Finite RNG-only qualification; all host reductions are outside capture."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", torch.cuda.current_device())
    checks = {}
    statistics = {}

    def check(name, passed):
        checks[name] = bool(passed)

    def bounded(name, observed, expected, tolerance, justification):
        statistics[name] = {
            "observed": float(observed), "expected": float(expected),
            "absolute_tolerance": float(tolerance), "justification": justification,
        }
        check(name, abs(observed - expected) <= tolerance)

    # Boundary counter words expose old flattened-uint32-offset collisions.
    seed = torch.tensor(-4611686018427387893, device=device, dtype=torch.int64)
    pairs = torch.tensor([0, 1, 65536, 16777216, 2147483648, 4294967295], device=device)
    positions = torch.tensor([4294967295, 2147483648, 16777216, 65536, 1, 0], device=device)
    original_seed, original_pairs, original_positions = seed.clone(), pairs.clone(), positions.clone()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()

    def draw(s, p, t):
        return coupled_rng.coupled_normal(s, p, t, dimension=7, role=1), coupled_rng.coupled_uniform(s, p, t, role=2), coupled_rng.coupled_uniform(s, p, t, role=3)

    base = draw(seed, pairs, positions)
    check("repeat_bitwise", all(torch.equal(a, b) for a, b in zip(base, draw(seed, pairs, positions))))
    check("fp32_gradient_free", all(x.dtype == torch.float32 and not x.requires_grad for x in base))
    permutation = torch.tensor([5, 2, 0, 4, 1, 3], device=device)
    reordered = draw(seed, pairs[permutation], positions[permutation])
    check("row_order_bitwise", all(torch.equal(a[permutation], b) for a, b in zip(base, reordered)))
    pad = torch.full((61,), -1, device=device, dtype=torch.int64)
    wider = draw(seed, torch.cat((pairs, pad)), torch.cat((positions, pad)))
    check("padded_width_bitwise", all(torch.equal(a, b[:pairs.numel()]) for a, b in zip(base, wider)))
    check("padding_zero", all(torch.count_nonzero(b[pairs.numel():]).item() == 0 for b in wider))
    wider_dimension = coupled_rng.coupled_normal(seed, pairs, positions, dimension=19, role=1)
    check("component_width_bitwise", torch.equal(base[0], wider_dimension[:, :7]))
    strided_pairs = torch.stack((pairs, pairs), dim=1)[:, 0]
    strided_positions = torch.stack((positions, positions), dim=1)[:, 1]
    check("strided_indices_bitwise", all(torch.equal(a, b) for a, b in zip(base, draw(seed, strided_pairs, strided_positions))))
    paired = draw(seed, pairs.repeat_interleave(2), positions.repeat_interleave(2))
    signs = torch.tensor([1.0, -1.0], device=device).repeat(pairs.numel())
    signed = paired[0] * signs[:, None]
    check("antithetic_exact_negation", torch.equal(signed[::2], -signed[1::2]))
    check("gate_common_random_numbers", torch.equal(paired[1][::2], paired[1][1::2]))
    check("answer_common_random_numbers", torch.equal(paired[2][::2], paired[2][1::2]))
    check("role_domains_differ", not torch.equal(base[1], base[2]))
    swapped = draw(seed, positions, pairs)
    check("pair_and_position_words_distinct", not torch.equal(base[0], swapped[0]))
    overflow = draw(seed, torch.tensor([4294967296], device=device), torch.zeros(1, device=device, dtype=torch.int64))
    check("overflow_not_silent_wrap", all(torch.isnan(x).all().item() for x in overflow))
    check("empty_batch", all(x.shape[0] == 0 for x in draw(seed, pairs[:0], positions[:0])))

    compiled = torch.compile(draw, fullgraph=True, dynamic=True)
    compiled_base = compiled(seed, pairs, positions)
    check("compile_fullgraph_bitwise", all(torch.equal(a, b) for a, b in zip(base, compiled_base)))
    def shape_addressed_draw(s, p, t, reference):
        return coupled_rng.coupled_normal(s, p, t, dimension=reference.shape[1], role=1)

    compiled_shape = torch.compile(shape_addressed_draw, fullgraph=True, dynamic=True)
    check("symbolic_component_width_bitwise",
          torch.equal(base[0], compiled_shape(seed, pairs, positions, base[0])))
    # Warm eager and compiled operators on a side stream before graph recording.
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            compiled(seed, pairs, positions)
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = compiled(seed, pairs, positions)
    graph.replay()
    check("capture_initial_bitwise", all(torch.equal(a, b) for a, b in zip(base, captured)))
    for iteration in range(3):
        # These copies preserve captured addresses while replacing their contents.
        seed.copy_(original_seed + iteration + 1)
        pairs.copy_(original_pairs.flip(0))
        positions.copy_((original_positions + iteration + 1) % 2**32)
        expected = draw(seed, pairs, positions)
        graph.replay()
        check(f"capture_dynamic_contents_{iteration}", all(torch.equal(a, b) for a, b in zip(expected, captured)))
        check(f"changed_counter_changes_draw_{iteration}", not torch.equal(base[0], captured[0]))
    seed.copy_(original_seed)
    pairs.copy_(original_pairs)
    positions.copy_(original_positions)
    graph.replay()
    check("capture_restored_bitwise", all(torch.equal(a, b) for a, b in zip(base, captured)))
    check("inputs_unchanged_by_draw", torch.equal(seed, original_seed) and torch.equal(pairs, original_pairs) and torch.equal(positions, original_positions))
    check("torch_cpu_rng_unchanged", torch.equal(cpu_rng, torch.get_rng_state()))
    check("torch_cuda_rng_unchanged", torch.equal(cuda_rng, torch.cuda.get_rng_state(device)))

    rows, dimension = 262144, 4
    sample_pairs = torch.arange(rows, device=device, dtype=torch.int64)
    sample_positions = torch.full_like(sample_pairs, 9713)
    z_gpu = coupled_rng.coupled_normal(seed, sample_pairs, sample_positions, dimension=dimension, role=1)
    u_gpu = coupled_rng.coupled_uniform(seed, sample_pairs, sample_positions, role=2)
    v_gpu = coupled_rng.coupled_uniform(seed, sample_pairs, sample_positions, role=3)
    z, u, v = z_gpu.cpu().double(), u_gpu.cpu().double(), v_gpu.cpu().double()
    flat = z.flatten()
    n = flat.numel()
    check("normal_all_finite", torch.isfinite(z).all())
    check("uniform_half_open_interval", ((u >= 0) & (u < 1) & (v >= 0) & (v < 1)).all())
    bound = math.sqrt(-2 * math.log(1e-7))
    check("normal_finite_precision_tail_bound", (flat.abs() <= bound + 2e-6).all())
    mean = flat.mean().item()
    centered = flat - mean
    variance = centered.square().mean().item()
    bounded("normal_mean", mean, 0, 7 / math.sqrt(n), "7 standard errors, SE=1/sqrt(N)")
    bounded("normal_variance", variance, 1, 7 * math.sqrt(2 / n), "7 asymptotic SE, sqrt(2/N); population variance estimator")
    bounded("normal_skewness", (centered.pow(3).mean() / variance**1.5).item(), 0, 7 * math.sqrt(6 / n), "7 asymptotic normal-reference SE, sqrt(6/N)")
    bounded("normal_excess_kurtosis", (centered.pow(4).mean() / variance**2 - 3).item(), 0, 7 * math.sqrt(24 / n), "7 asymptotic normal-reference SE, sqrt(24/N)")
    bounded("uniform_mean", u.mean().item(), 0.5, 7 / math.sqrt(12 * rows), "7 SE for U[0,1), 1/sqrt(12*N)")
    bounded("uniform_variance", u.var(unbiased=False).item(), 1 / 12, 7 / math.sqrt(180 * rows), "7 asymptotic SE for uniform variance, 1/sqrt(180*N)")
    for threshold in (1, 2, 3):
        probability = math.erfc(threshold / math.sqrt(2))
        observed = (flat.abs() > threshold).double().mean().item()
        bounded(f"normal_two_sided_tail_{threshold}", observed, probability, 7 * math.sqrt(probability * (1 - probability) / n), "7 binomial SE using normal-reference tail probability")
    # DKW has no histogram/bin-selection ambiguity. Applied under PRNG's iid model.
    alpha = 1e-6
    sorted_u = u.sort().values
    ranks = torch.arange(1, rows + 1, dtype=torch.float64) / rows
    ks = max((ranks - sorted_u).max().item(), (sorted_u - (ranks - 1 / rows)).max().item())
    bounded("uniform_KS_distance", ks, 0, math.sqrt(math.log(2 / alpha) / (2 * rows)), "DKW bound under iid reference, per-check alpha=1e-6")
    for name, a, b in (("normal_component_correlation", z[:, 0], z[:, 1]), ("uniform_role_correlation", u, v), ("gaussian_gate_correlation", z[:, 0], u), ("adjacent_pair_correlation", z[:-1, 0], z[1:, 0])):
        a, b = a - a.mean(), b - b.mean()
        corr = (a * b).sum() / (a.square().sum() * b.square().sum()).sqrt()
        bounded(name, corr.item(), 0, 7 / math.sqrt(a.numel()), "7 asymptotic null SE for correlation, 1/sqrt(N)")
    return {
        "schema": "minicpm-coupled-rng-qualification/v1",
        "scope": "RNG qualification only; no learning, variance-reduction, or optimizer-adoption evidence",
        "passed": all(checks.values()), "checks": checks, "statistics": statistics,
        "sample_size": {"normal": n, "uniform_per_role": rows, "normal_components": dimension},
        "tolerance_policy": "7-SE normal-reference screening plus DKW alpha=1e-6; approximate iid PRNG assumptions, not an exact distribution or independence proof",
        "metadata": coupled_rng.rng_metadata(), "device": torch.cuda.get_device_name(device),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queued-run", action="store_true", required=True,
                        help="Attest this foreground process is supervised by mlq; not a queue bypass.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Refuse clobbering before allocating or launching any CUDA work.
    with args.output.open("x") as output:
        try:
            report = _qualify()
        except Exception as exc:
            report = {"schema": "minicpm-coupled-rng-qualification/v1", "passed": False,
                      "error": f"{type(exc).__name__}: {exc}", "metadata": coupled_rng.rng_metadata()}
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
            raise
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}))
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
