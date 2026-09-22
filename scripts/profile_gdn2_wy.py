"""One production-shaped original-FLA WY backward for queued Nsight profiling.

Run under ncu --profile-from-start off. Three complete kernel-only backward
warmups resolve autotuning; CUDA profiler start/stop encloses one more backward.
The ncu kernel filter selects only WY. Synthetic tensors diagnose execution,
not model quality, optimizer performance, or complete-update throughput.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
from fla.ops.gdn2.chunk import chunk_gdn2
from fla.ops.gdn2.chunk_bwd import chunk_gdn2_bwd_kernel_wy_dqkg_fused


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required; submit through mlq")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(733)
    shape = (64, 1024, 4, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q) * .2
    g = -.01 - torch.rand(shape, device="cuda", dtype=torch.float32) * .5
    b = torch.rand_like(q)
    w = torch.rand_like(q)
    leaves = [tensor.detach().requires_grad_() for tensor in (q, k, v, g, b, w)]
    upstream = torch.randn_like(v)

    def forward():
        for tensor in leaves:
            tensor.grad = None
        output, _ = chunk_gdn2(*leaves, use_qk_l2norm_in_kernel=True,
                              initial_state=None, output_final_state=False,
                              state_v_first=False, disable_recompute=True)
        return output

    for _ in range(3):
        forward().backward(upstream)
    torch.cuda.synchronize()
    # Heuristics wraps FLA's CachedAutotuner; report the selected actual tile.
    tuner = chunk_gdn2_bwd_kernel_wy_dqkg_fused.fn
    config = tuner.best_config
    report = dict(status="prepared", operation="original_fla_gdn2_wy_backward",
                  shape=list(shape), state_v_first=False, disable_recompute=True,
                  initial_state=False, final_state=False, qkv_dtype="bfloat16",
                  gate_dtype="float32", erase_write_dtype="bfloat16",
                  gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  selected_config=dict(kwargs=config.kwargs, num_warps=config.num_warps,
                                       num_stages=config.num_stages, num_ctas=config.num_ctas),
                  warmup_backwards=3, profiled_backwards=1,
                  limitations="Synthetic operator inputs; ncu replay perturbs timing; no model or update speed claim",
                  source_sha256={})
    for path in (Path(__file__), Path(inspect.getfile(inspect.unwrap(chunk_gdn2))),
                 Path(inspect.getfile(type(tuner))),
                 Path(inspect.getfile(__import__('fla.ops.gdn2.chunk_bwd', fromlist=[''])))):
        report['source_sha256'][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    output = forward()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    try:
        output.backward(upstream)
        torch.cuda.synchronize()
    finally:
        torch.cuda.cudart().cudaProfilerStop()
    report['finite_gradients'] = {name: bool(tensor.grad is not None and torch.isfinite(tensor.grad).all())
                                  for name, tensor in zip(('q', 'k', 'v', 'g', 'b', 'w'), leaves)}
    if not all(report['finite_gradients'].values()):
        report['status'] = 'failed'
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        raise FloatingPointError('Nonfinite or missing operator gradient')
    report['status'] = 'completed'
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
