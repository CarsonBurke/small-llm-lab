#!/usr/bin/env python3
"""Queue an exclusive real-logit sampler comparison; never changes production defaults.

The bounded-tie candidate is deliberately experimental: it cannot preserve all
kth-score ties if support exceeds its static capacity, and HF's unstable full
sort can choose different token labels at a tied nucleus boundary. Both failure
modes are reported, not hidden behind a claim of HF compatibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKER_TOKEN_ENV = "PARAMETER_GOLF_SAMPLER_BENCHMARK_TOKEN"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--logits', type=Path, default=ROOT / 'ablation_results/minicpm_low_noise_20260911/official_inference_parity_logits.pt')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--temperature', type=float, default=0.9)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--tie-capacity-factor', type=int, default=4)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--queue-token', help=argparse.SUPPRESS)
    return p


def filtered_support(logits, *, top_k, temperature, top_p, method, capacity_factor):
    """Benchmark-only candidates; return compact probabilities and token labels."""
    capacity = min(logits.shape[-1], top_k * capacity_factor) if method == 'bounded_ties' else top_k
    values, ids = logits.topk(capacity, dim=-1, sorted=True)
    scores = values.float() / temperature
    if method == 'bounded_ties':
        scores = scores.masked_fill(values < values[:, top_k - 1:top_k], -torch.inf)
    if method == 'legacy':
        probabilities = scores.softmax(-1)
        if top_p < 1:
            preceding = probabilities.cumsum(-1) - probabilities
            probabilities = torch.where(preceding < top_p, probabilities, 0.0)
            probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    else:
        # HF removes the LOWEST scores with cumulative probability <= 1-p,
        # preserving at least one token, then softmaxes the surviving scores.
        # Only the compact support is sorted; no full-vocabulary sort here.
        scores, order = scores.sort(dim=-1)
        ids = ids.gather(1, order)
        if top_p < 1:
            remove = scores.softmax(-1).cumsum(-1) <= (1 - top_p)
            remove[:, -1] = False
            scores = scores.masked_fill(remove, -torch.inf)
        probabilities = scores.softmax(-1)
    return probabilities, ids


def sample_support(probabilities, ids, uniform=None):
    if uniform is None:
        return ids.gather(1, torch.multinomial(probabilities, 1)).squeeze(1)
    ids, order = ids.sort(dim=-1)
    probabilities = probabilities.gather(1, order)
    mass = probabilities.double().cumsum(-1)
    mass = mass / mass[:, -1:]
    ranks = torch.arange(probabilities.shape[-1], device=probabilities.device)
    last_positive = torch.where(probabilities > 0, ranks, -1).amax(-1)
    mass = torch.where(ranks[None] >= last_positive[:, None], 1.0, mass)
    selected = (uniform.double()[:, None] >= mass).sum(-1)
    selected = torch.minimum(selected, last_positive)
    return ids.gather(1, selected[:, None]).squeeze(1)


def agreement(candidate, reference):
    difference = (candidate - reference).abs()
    return {
        'tv_mean': float((difference.sum(-1) / 2).mean()),
        'tv_max': float((difference.sum(-1) / 2).max()),
        'max_probability_error': float(difference.max()),
        'support_mismatch_rows': int(((candidate > 0) != (reference > 0)).any(-1).sum()),
        'rows': candidate.shape[0],
    }


def qualify(logits, hf, args, top_k):
    reference = hf(logits)
    values = logits.topk(top_k, dim=-1).values
    cutoff = values[:, -1:]
    finite = torch.isfinite(logits)
    tie_count = ((logits == cutoff) & finite).sum(-1)
    support = ((logits >= cutoff) & finite).sum(-1)
    capacity = min(logits.shape[-1], top_k * args.tie_capacity_factor)
    report = {
        'kth_score_tied_rows': int((tie_count > 1).sum()),
        'hf_topk_support_max': int(support.max()),
        'hf_topk_expands_past_k_rows': int((support > top_k).sum()),
        'bounded_ties_overflow_rows': int((support > capacity).sum()),
        'bounded_ties_capacity': capacity,
        'methods': {},
    }
    for method in ('legacy', 'ascending_exact_k', 'bounded_ties'):
        probabilities, ids = filtered_support(logits, top_k=top_k, temperature=args.temperature,
            top_p=args.top_p, method=method, capacity_factor=args.tie_capacity_factor)
        dense = torch.zeros_like(reference).scatter(1, ids, probabilities)
        report['methods'][method] = agreement(dense, reference)
    return report


def graph_timing(call, static_logits, real_logits, args):
    # Each method closes over different static sampling controls. Isolate their
    # compilation caches rather than exhausting Dynamo's per-code guard limit.
    torch._dynamo.reset()
    compiled = torch.compile(call, fullgraph=True, mode='default')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(args.warmup):
            compiled(static_logits)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = compiled(static_logits)
    results = {}
    for position in (0, 8, 16):
        static_logits.copy_(real_logits[:, position].repeat_interleave(static_logits.shape[0] // 4, dim=0))
        for _ in range(args.warmup):
            graph.replay()
        measurements = []
        for _ in range(args.repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                graph.replay()
            end.record()
            end.synchronize()
            measurements.append(start.elapsed_time(end) * 1000 / args.iterations)
        results[str(position)] = {'median_us': statistics.median(measurements),
            'min_us': min(measurements), 'max_us': max(measurements), 'repeats_us': measurements}
    # Retain graph outputs until all replays and event measurements finish.
    del output, graph, compiled
    return results


def worker(args):
    if not args.queue_token or os.environ.get(WORKER_TOKEN_ENV) != args.queue_token:
        raise RuntimeError('execute through mlq using this script without --worker')
    global torch
    import torch
    import transformers
    from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper
    sys.path.insert(0, str(ROOT))
    from postraining.fast_inference import selected_token_logprobs, top_k_top_p_sample
    from postraining.minicpm_paired_rollout import _fixed_label_categorical

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = torch.load(args.logits, map_location='cpu', weights_only=True)
    actual = payload['actual']
    if tuple(actual.shape) != (4, 17, 130560):
        raise ValueError(f'expected real logits [4,17,130560], got {tuple(actual.shape)}')
    references = payload['reference']
    if not isinstance(references, torch.Tensor):
        references = torch.stack(references)
    if references.shape != actual.shape:
        raise ValueError('reference logits do not match real-logit shape')
    actual = actual.to(device='cuda', dtype=torch.bfloat16)
    references = references.to(device='cuda', dtype=torch.bfloat16)
    torch.manual_seed(1337)
    report = {'schema': 'minicpm-sampler-comparison/v1', 'status': 'running',
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'device': torch.cuda.get_device_name(), 'dtype': str(actual.dtype),
        'logits_sha256': hashlib.sha256(args.logits.read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'temperature': args.temperature, 'top_p': args.top_p,
        'iterations': args.iterations, 'repeats': args.repeats, 'warmup': args.warmup,
        'production_changed': False, 'qualification': {}, 'timing': {},
        'caveats': ['Microbenchmark only; not end-to-end decode throughput.',
            'Bounded ties is not generally HF-compatible: overflow and tied nucleus label order can differ.',
            'Reference logits are cast to BF16 to match the production head input.',
            'CUDA graphs replay saved real logits at positions 0,8,16; four prompt rows replicated to B16/B64.']}
    boundary_args = argparse.Namespace(**{**vars(args), 'temperature': 1.0, 'top_p': 0.5})
    boundary_logits = torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.5, 0.25, 0.125, 0.125]], device='cuda').log()
    boundary_topk, boundary_topp = TopKLogitsWarper(4), TopPLogitsWarper(0.5)
    report['nucleus_half_mass_boundary'] = qualify(
        boundary_logits,
        lambda logits: boundary_topp(None, boundary_topk(None, logits)).softmax(-1),
        boundary_args, 4,
    )
    for top_k in (20, 50):
        topk_warper, topp_warper = TopKLogitsWarper(top_k), TopPLogitsWarper(args.top_p)

        def hf(logits):
            scores = logits.float() / args.temperature
            return topp_warper(None, topk_warper(None, scores)).softmax(-1)

        qualification = {name: qualify(tensor.flatten(0, 1), hf, args, top_k)
            for name, tensor in (('actual', actual), ('reference', references))}
        # Explicitly expose both bounded-support overflow and nucleus equality.
        # These diagnostics are not performance inputs and do not replace real logits.
        fixtures = torch.full((3, 130560), -torch.inf, device='cuda')
        fixtures[0, :top_k + 1] = 0
        fixtures[1, :top_k * args.tie_capacity_factor + 1] = 0
        fixtures[2, :4] = 0
        qualification['tie_and_boundary_fixtures'] = qualify(fixtures, hf, args, top_k)
        report['qualification'][str(top_k)] = qualification
        for batch_size in (64,):
            static_logits = actual[:, 0].repeat_interleave(batch_size // 4, dim=0).contiguous()
            uniform = torch.linspace(0.01, 0.99, batch_size, device='cuda')
            for paired in (False, True):
                def call(logits):
                    if paired:
                        token = _fixed_label_categorical(logits, uniform,
                            temperature=args.temperature, top_k=top_k, top_p=args.top_p)
                    else:
                        token = top_k_top_p_sample(logits, temperature=args.temperature,
                            top_k=top_k, top_p=args.top_p)
                    return token, selected_token_logprobs(logits, token)

                key = f'legacy_k{top_k}_b{batch_size}_{"paired" if paired else "independent"}'
                report['timing'][key] = graph_timing(call, static_logits, actual, args)
                print(json.dumps({'finished': key, 'timing': report['timing'][key]}), flush=True)
    report['status'] = 'completed'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


def main():
    args = parser().parse_args()
    if min(args.iterations, args.repeats, args.warmup, args.tie_capacity_factor) <= 0:
        raise ValueError('iteration, repeat, warmup, and capacity counts must be positive')
    if not math.isfinite(args.temperature) or args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError('invalid sampling temperature/top-p')
    args.logits, args.output = args.logits.resolve(), args.output.resolve()
    if not args.logits.is_file():
        raise FileNotFoundError(args.logits)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.worker:
        return worker(args)
    token = secrets.token_hex(16)
    command = ['mlq', 'submit', '--name', 'minicpm-sampler-comparison', '--max-parallel-runs', '1',
        '--time-limit', '5m', '--max-attempts', '1', '--cwd', str(ROOT),
        '--env', f'{WORKER_TOKEN_ENV}={token}', '--', sys.executable, str(Path(__file__).resolve()),
        '--worker', '--queue-token', token, '--logits', str(args.logits), '--output', str(args.output),
        '--iterations', str(args.iterations), '--repeats', str(args.repeats), '--warmup', str(args.warmup),
        '--temperature', str(args.temperature), '--top-p', str(args.top_p),
        '--tie-capacity-factor', str(args.tie_capacity_factor)]
    return subprocess.run(command, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
