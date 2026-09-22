"""Queued original-FLA WY tile experiment, without changing global dispatch.

Build production-shaped operands through the original forward/backward helpers,
then call the original WY JIT directly with explicit tiles. This measures one
operator, not model throughput. Precision and arithmetic source stay unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import statistics

import torch
from fla.modules.l2norm import l2norm_fwd
from fla.ops.gdn2.chunk_fwd import chunk_gdn2_fwd
from fla.ops.gdn2.chunk_bwd import chunk_gdn2_bwd_wy_dqkg_fused, chunk_gdn2_bwd_kernel_wy_dqkg_fused
from fla.ops.kda.chunk_bwd import chunk_kda_bwd_dAv
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu


def config_dict(config):
    return dict(BK=config.kwargs['BK'], BV=config.kwargs['BV'],
                num_warps=config.num_warps, num_stages=config.num_stages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--warp-probes', action='store_true',
                        help='Test thread layouts for the measured register-pressure bottleneck')
    args = parser.parse_args()
    if args.repeats < 10:
        parser.error('At least ten independent measurements required')
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 required; submit through mlq')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', purpose='isolated_original_WY_tile_measurement',
                  shape=[64, 1024, 4, 128, 128], state_v_first=False, disable_recompute=True,
                  gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  repeats=args.repeats, warmups=5, parity_relative_norm_bound=1e-4,
                  limitations='Synthetic operands; compiled resource counts are not measured hardware occupancy; no full-update speed claim',
                  cases=[], source_sha256={})
    for function in (chunk_gdn2_fwd, chunk_gdn2_bwd_wy_dqkg_fused,
                     chunk_kda_bwd_dAv, chunk_gated_delta_rule_bwd_dhu):
        path = Path(inspect.getfile(inspect.unwrap(function)))
        report['source_sha256'][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    report['source_sha256'][str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    save()
    try:
        torch.manual_seed(733)
        shape = (64, 1024, 4, 128)
        q, _ = l2norm_fwd(torch.randn(shape, device='cuda', dtype=torch.bfloat16))
        k, _ = l2norm_fwd(torch.randn_like(q))
        v = torch.randn_like(q) * .2
        raw_g = -.01 - torch.rand(shape, device='cuda', dtype=torch.float32) * .5
        b, w = torch.rand_like(q), torch.rand_like(q)
        scale = 128 ** -.5
        _, _, g, Aqk, A, wy_w, _, qg, kg, v_new, h, _ = chunk_gdn2_fwd(
            q=q, k=k, v=v, g=raw_g, b=b, w_gate=w, scale=scale,
            initial_state=None, output_final_state=False,
            disable_recompute=True, state_v_first=False)
        do = torch.randn_like(v)
        _, dv = chunk_kda_bwd_dAv(q=q, k=k, v=v_new, do=do, A=Aqk, scale=scale)
        dh, _, dv = chunk_gated_delta_rule_bwd_dhu(q=qg, k=kg, w=wy_w, do=do, dv=dv,
                                                   gk=g, scale=scale, state_v_first=False)
        operands = dict(q=q, k=k, v=v, v_new=v_new, g=g, b=b, w_gate=w,
                        A=A, h=h, do=do, dh=dh, dv=dv, scale=scale)
        expected = chunk_gdn2_bwd_wy_dqkg_fused(**operands)
        tuner = chunk_gdn2_bwd_kernel_wy_dqkg_fused.fn
        jit = tuner.fn
        baseline = config_dict(tuner.best_config)
        report['autotuned_baseline'] = baseline
        names = ('dq', 'dk', 'dv2', 'db', 'dw', 'dg', 'dA')
        outputs = dict(zip(names, (torch.empty_like(tensor) for tensor in expected)))
        launch_args = dict(**operands, **outputs, cu_seqlens=None, chunk_indices=None,
                           T=1024, H=4, K=128, V=128, BT=64,
                           STATE_V_FIRST=False, IS_VARLEN=False)

        def launch(config):
            return jit[(16, 256)](**launch_args, **config)

        def parity():
            errors = {}
            for name, reference in zip(names, expected):
                actual = outputs[name]
                if not bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()):
                    raise FloatingPointError(name)
                error = float((actual.double() - reference.double()).norm()
                              / reference.double().norm().clamp_min(1e-12))
                errors[name] = error
            return errors

        def measure(config):
            for _ in range(5):
                launch(config)
            torch.cuda.synchronize()
            samples = []
            for _ in range(args.repeats):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                launch(config)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            return dict(samples_ms=samples, median_ms=statistics.median(samples))

        def resource(kernel):
            return dict(n_regs=kernel.n_regs, n_spills=kernel.n_spills,
                        shared_bytes=kernel.metadata.shared,
                        num_warps=kernel.metadata.num_warps,
                        num_stages=kernel.metadata.num_stages,
                        default_dot_input_precision=kernel.metadata.default_dot_input_precision,
                        enable_fp_fusion=kernel.metadata.enable_fp_fusion,
                        compiled_hash=kernel.hash)

        compiled = launch(baseline)
        baseline_errors = parity()
        if max(baseline_errors.values()) >= 1e-4:
            raise AssertionError(f'Direct baseline differs from wrapper: {baseline_errors}')
        report['baseline_resources'] = resource(compiled)
        report['baseline_parity'] = baseline_errors
        report['baseline_timing'] = measure(baseline)
        save()
        # Three directional probes, not a broad autotuning sweep. Existing
        # warp count and stages stay fixed; only BK/BV exceed the old search.
        probes = ([(32, 32, 8), (32, 32, 16), (32, 128, 8), (64, 64, 8)]
                  if args.warp_probes else [(128, baseline['BV'], baseline['num_warps']),
                                           (baseline['BK'], 128, baseline['num_warps']),
                                           (128, 128, baseline['num_warps'])])
        for bk, bv, warps in probes:
            config = dict(baseline, BK=bk, BV=bv, num_warps=warps)
            case = dict(config=config)
            report['cases'].append(case)
            try:
                kernel = launch(config)
                case['resources'] = resource(kernel)
                case['relative_errors'] = parity()
                case['parity_passed'] = max(case['relative_errors'].values()) < 1e-4
                if case['parity_passed']:
                    case['control_before'] = measure(baseline)
                    case['candidate'] = measure(config)
                    case['control_after'] = measure(baseline)
                    control_ms = (case['control_before']['median_ms'] + case['control_after']['median_ms']) / 2
                    case['kernel_speedup'] = control_ms / case['candidate']['median_ms']
                    case['timings_separated'] = max(case['candidate']['samples_ms']) < min(
                        case['control_before']['samples_ms'] + case['control_after']['samples_ms'])
                case['status'] = 'qualified' if case['parity_passed'] else 'rejected_numerical_difference'
            except Exception as error:
                case.update(status='failed', error=f'{type(error).__name__}: {error}')
            save()
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
