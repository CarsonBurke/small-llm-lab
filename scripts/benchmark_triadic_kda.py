"""Cost of triadic KDA state slices against the production KDA chunk kernel.

Times one mixer recurrence, forward plus backward, at the KDA8 training
shape for FLA's ``chunk_kda`` (the kernel every KDA run uses) and for
``triadic_chunk_kda`` at each slice count. ``E = 1`` isolates what the plain
PyTorch chunk form costs against the fused Triton kernels; larger ``E`` adds
what the extra state costs on top. It also checks the two agree at ``E = 1``
on the GPU, in the kernel's own precision.

    python scripts/benchmark_triadic_kda.py --out <report.json>

It executes on the GPU, so it runs through ``mlq``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from pretraining.nanogpt_mini.nanogpt_mini_kda_model import (
    KDA_SAFE_GATE_LOWER_BOUND,
    _l2norm,
    kda_decay_gate,
)
from pretraining.nanogpt_mini.triadic_kda import triadic_chunk_kda


def raw_inputs(batch, tokens, heads, width, slices, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(
            *shape, generator=generator, device="cuda", dtype=torch.float32
        ).to(dtype).requires_grad_()

    return {
        "q": randn(batch, tokens, heads, width),
        "k": randn(batch, tokens, heads, width),
        "v": randn(batch, tokens, heads, width),
        "decay": randn(batch, tokens, heads, width),
        "beta": randn(batch, tokens, heads, dtype=torch.float32),
        "q2": randn(batch, tokens, heads, slices, dtype=torch.float32),
        "k2": randn(batch, tokens, heads, slices, dtype=torch.float32),
        "slice_decay": randn(batch, tokens, heads, slices, dtype=torch.float32),
    }


def run_fla(x, A_log, dt_bias):
    from fla.ops.kda import chunk_kda

    o, _ = chunk_kda(
        q=x["q"], k=x["k"], v=x["v"], g=x["decay"], beta=x["beta"],
        A_log=A_log, dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True, safe_gate=True,
        lower_bound=KDA_SAFE_GATE_LOWER_BOUND, state_v_first=True,
        disable_recompute=False,
    )
    return o


def run_triadic(x, A_log, dt_bias, slice_A_log, slice_dt_bias, single_slice):
    width = x["q"].size(-1)
    if single_slice:
        ones = torch.ones_like(x["q2"][..., :1])
        q2, k2, slice_log_decay = ones, ones, torch.zeros_like(ones)
    else:
        q2 = _l2norm(F.softplus(x["q2"]))
        k2 = _l2norm(F.softplus(x["k2"]))
        slice_log_decay = kda_decay_gate(x["slice_decay"], slice_A_log, slice_dt_bias)
    o, _ = triadic_chunk_kda(
        _l2norm(x["q"].float()) * width**-0.5,
        _l2norm(x["k"].float()),
        x["v"].float(),
        q2, k2,
        kda_decay_gate(x["decay"], A_log, dt_bias),
        slice_log_decay,
        torch.sigmoid(x["beta"]),
    )
    return o


def time_forward_backward(fn, inputs, repeats):
    """Median seconds and peak bytes of forward + backward to the inputs."""
    leaves = list(inputs.values())
    samples = []
    torch.cuda.reset_peak_memory_stats()
    for index in range(repeats + 2):  # two warm-up rounds
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn(inputs).float().square().sum().backward()
        torch.cuda.synchronize()
        if index >= 2:
            samples.append(time.perf_counter() - start)
        for leaf in leaves:
            leaf.grad = None
    return statistics.median(samples), torch.cuda.max_memory_allocated()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--slices", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--shapes", type=int, nargs="+", default=[32, 1024, 8, 5120],
        help="batch tokens pairs",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out_path = Path(args.out)
    if out_path.exists():
        parser.error(f"{out_path} exists; reports are not overwritten")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    H, D = args.heads, args.width
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    A_log = torch.zeros(H, device="cuda")
    dt_bias = torch.randn(H * D, device="cuda", generator=generator) - 4.0
    report = {
        "schema": "benchmark_triadic_kda/v1",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "heads": H,
        "width": D,
        "repeats": args.repeats,
        "shapes": [],
    }
    for batch, tokens in zip(args.shapes[::2], args.shapes[1::2], strict=True):
        entry = {"batch": batch, "tokens": tokens, "variants": {}}
        x = raw_inputs(batch, tokens, H, D, 1, args.seed)
        with torch.no_grad():
            reference = run_fla(x, A_log, dt_bias).float()
            ours = run_triadic(x, A_log, dt_bias, None, None, single_slice=True)
            entry["single_slice_relative_error"] = float(
                (ours - reference).norm() / reference.norm()
            )
        seconds, peak = time_forward_backward(
            lambda inputs: run_fla(inputs, A_log, dt_bias), x, args.repeats
        )
        entry["variants"]["fla_chunk_kda"] = {"seconds": seconds, "peak_bytes": peak}
        baseline = seconds
        for slices in args.slices:
            x = raw_inputs(batch, tokens, H, D, slices, args.seed)
            slice_A_log = torch.zeros(H, device="cuda")
            slice_dt_bias = torch.full((H * slices,), -6.0, device="cuda")
            try:
                seconds, peak = time_forward_backward(
                    lambda inputs: run_triadic(
                        inputs, A_log, dt_bias, slice_A_log, slice_dt_bias,
                        single_slice=slices == 1,
                    ),
                    x, args.repeats,
                )
            except torch.OutOfMemoryError:
                entry["variants"][f"triadic_e{slices}"] = {"out_of_memory": True}
                torch.cuda.empty_cache()
                continue
            entry["variants"][f"triadic_e{slices}"] = {
                "seconds": seconds,
                "peak_bytes": peak,
                "ratio_to_fla": seconds / baseline,
            }
        report["shapes"].append(entry)
        print(json.dumps(entry, indent=2), flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
