"""Production-shape correctness and operator latency for KDA and GDN-2.

The suite runs each backend in a fresh subprocess because FLA caches backend
availability and selection at import time. It compares the exact
``B=8,T=1024,H=3,D=128`` operators used by the GPT-2-vocab ablations,
including every differentiable input. GDN-2's fused decay activation is
checked against the NVIDIA reference's externally evaluated decay equation.
The suite reports steady-state forward+backward latency after backend
compilation and autotuning.

This is a GPU/model workload and must only be launched through ``mlq``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


CONFIGS = (
    ("gdn2_external_gate_v_first", "gdn2", "0", False, True, True),
    ("gdn2_recompute_v_first", "gdn2", "0", False, True, False),
    ("gdn2_no_recompute_v_first", "gdn2", "0", True, True, False),
    ("gdn2_external_gate_k_first", "gdn2", "0", False, False, True),
    ("gdn2_recompute_k_first", "gdn2", "0", False, False, False),
    ("gdn2_no_recompute_k_first", "gdn2", "0", True, False, False),
    ("kda_triton_recompute", "kda", "0", False, True, False),
    ("kda_triton_no_recompute", "kda", "0", True, True, False),
    ("kda_tilelang_recompute", "kda", "1", False, True, False),
    ("kda_tilelang_no_recompute", "kda", "1", True, True, False),
)

KDA_TOLERANCES = {
    "output": 0.005,
    "q_grad": 0.008,
    "k_grad": 0.008,
    "v_grad": 0.008,
    "g_grad": 0.02,
    "beta_grad": 0.02,
    "A_log_grad": 0.003,
    "dt_bias_grad": 0.008,
}

GDN2_TOLERANCES = {
    "output": 0.005,
    "q_grad": 0.01,
    "k_grad": 0.01,
    "v_grad": 0.01,
    "g_grad": 0.02,
    "b_logits_grad": 0.02,
    "w_logits_grad": 0.02,
    "A_log_grad": 0.02,
    "dt_bias_grad": 0.02,
}


def _run_worker(args: argparse.Namespace) -> None:
    # Backend selection is cached while FLA imports, so set it first.
    os.environ["FLA_DISABLE_BACKEND_DISPATCH"] = "0"
    os.environ["FLA_FLASH_KDA"] = "0"
    os.environ["FLA_TILELANG"] = args.tilelang

    import torch
    import fla
    import tilelang
    import triton
    from fla.ops.gdn2 import chunk_gdn2
    from fla.ops.kda import chunk_kda
    from fla.ops.kda.backends import kda_registry
    from fla.utils import find_spec_cached, has_usable_nvcc

    if not torch.cuda.is_available():
        raise RuntimeError("KDA benchmark requires CUDA")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    shape = (args.batch_size, args.seq_len, args.num_heads, args.head_dim)

    def make_case(seed: int):
        torch.manual_seed(seed)
        leaves = {
            name: torch.randn(
                shape,
                device=device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            for name in ("q", "k", "v", "g")
        }
        if args.op == "gdn2":
            A_log = torch.empty(
                args.num_heads, device=device, dtype=torch.float32
            ).uniform_(1, 16).log()
        else:
            A_log = torch.zeros(
                args.num_heads, device=device, dtype=torch.float32
            )
        leaves["A_log"] = A_log.requires_grad_()
        log_dt = torch.empty(
            args.num_heads * args.head_dim, device=device, dtype=torch.float32
        ).uniform_(math.log(0.001), math.log(0.1))
        dt = log_dt.exp().clamp(min=1e-4)
        leaves["dt_bias"] = (
            dt + torch.log(-torch.expm1(-dt))
        ).requires_grad_()
        output_grad = torch.randn(shape, device=device, dtype=torch.bfloat16)
        if args.op == "kda":
            leaves["beta"] = torch.randn(
                shape[:-1],
                device=device,
                dtype=torch.float32,
                requires_grad=True,
            )
        else:
            leaves["b_logits"] = torch.randn(
                shape,
                device=device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            leaves["w_logits"] = torch.randn(
                shape,
                device=device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
        return leaves, output_grad

    def clear_gradients(leaves) -> None:
        for tensor in leaves.values():
            tensor.grad = None

    def forward_backward(leaves, output_grad):
        use_gate_in_kernel = not args.external_gate
        gate_input = leaves["g"]
        if args.external_gate:
            gate_input = (
                -leaves["A_log"].float().exp()[None, None, :, None]
                * torch.nn.functional.softplus(
                    leaves["g"].float()
                    + leaves["dt_bias"].view(
                        1,
                        1,
                        args.num_heads,
                        args.head_dim,
                    )
                )
            )
        common = dict(
            q=leaves["q"],
            k=leaves["k"],
            v=leaves["v"],
            g=gate_input,
            A_log=leaves["A_log"],
            dt_bias=leaves["dt_bias"],
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=use_gate_in_kernel,
            state_v_first=args.state_v_first,
            disable_recompute=args.disable_recompute,
        )
        if args.op == "kda":
            output, _ = chunk_kda(
                beta=leaves["beta"],
                use_beta_sigmoid_in_kernel=True,
                safe_gate=True,
                lower_bound=-5.0,
                **common,
            )
        else:
            output, _ = chunk_gdn2(
                b=leaves["b_logits"].sigmoid(),
                w=leaves["w_logits"].sigmoid(),
                safe_gate=False,
                **common,
            )
        output.backward(output_grad)
        return output

    # Untimed passes supply correctness artifacts for several independent
    # inputs. The first also initializes and autotunes the selected backend.
    artifacts = {}
    timing_leaves = None
    timing_output_grad = None
    for seed in range(args.seed, args.seed + args.parity_seeds):
        leaves, output_grad = make_case(seed)
        output = forward_backward(leaves, output_grad)
        artifacts[str(seed)] = {"output": output.detach().float().cpu()}
        artifacts[str(seed)].update({
            f"{name}_grad": tensor.grad.detach().float().cpu()
            for name, tensor in leaves.items()
        })
        clear_gradients(leaves)
        if timing_leaves is None:
            timing_leaves = leaves
            timing_output_grad = output_grad

    assert timing_leaves is not None and timing_output_grad is not None
    # Drop the final loop case if it is not the first retained timing case.
    leaves = output_grad = output = None
    dispatch_records = sorted(kda_registry._logged)
    tilelang_dispatch = any(
        record == "kda:chunk_kda_bwd_wy_dqkg_fused:tilelang"
        for record in dispatch_records
    )
    if args.op == "kda" and args.tilelang == "1" and not tilelang_dispatch:
        raise RuntimeError(
            "TileLang was requested but the KDA backward stage did not dispatch "
            f"to it; registry records: {dispatch_records}"
        )
    if args.tilelang == "0" and tilelang_dispatch:
        raise RuntimeError("TileLang dispatch occurred despite FLA_TILELANG=0")

    for _ in range(args.warmup):
        forward_backward(timing_leaves, timing_output_grad)
        clear_gradients(timing_leaves)
    torch.cuda.synchronize()

    memory_before_timing = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        forward_backward(timing_leaves, timing_output_grad)
        end.record()
        clear_gradients(timing_leaves)
    torch.cuda.synchronize()

    times_ms = [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]
    active_backend = kda_registry.get_active()
    result = {
        "requested_backend": "tilelang" if args.tilelang == "1" else "triton",
        "operation": args.op,
        "state_v_first": args.state_v_first,
        "active_backend": (
            active_backend.backend_type if active_backend is not None else "default"
        ),
        "tilelang_installed": find_spec_cached("tilelang") is not None,
        "usable_nvcc": has_usable_nvcc(),
        "tilelang_kda_backward_dispatched": tilelang_dispatch,
        "dispatch_records": dispatch_records,
        "disable_recompute": args.disable_recompute,
        "external_gate": args.external_gate,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "triton_version": triton.__version__,
        "fla_version": fla.__version__,
        "tilelang_version": tilelang.__version__,
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.fmean(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "timing_peak_delta_mib": (
            torch.cuda.max_memory_allocated(device) - memory_before_timing
        ) / 2**20,
        "times_ms": times_ms,
    }
    torch.save(artifacts, args.artifact)
    Path(args.result).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


def _error_ratio(reference, candidate) -> float:
    import torch

    error = (reference - candidate).flatten().square().mean().sqrt().item()
    scale = reference.flatten().square().mean().sqrt().item()
    return error / (scale + 1e-8)


def _run_suite(args: argparse.Namespace) -> None:
    import torch

    script = Path(__file__).resolve()
    selected_configs = tuple(
        config
        for config in CONFIGS
        if not args.kda_only or config[1] == "kda"
    )
    with tempfile.TemporaryDirectory(prefix="kda_training_bench_") as temp_dir:
        output_dir = Path(temp_dir)
        results = {}
        artifacts = {}

        for (
            name,
            operation,
            tilelang,
            disable_recompute,
            state_v_first,
            external_gate,
        ) in selected_configs:
            artifact_path = output_dir / f"{name}.pt"
            result_path = output_dir / f"{name}.json"
            command = [
                sys.executable,
                str(script),
                "--worker",
                "--tilelang",
                tilelang,
                "--op",
                operation,
                "--artifact",
                str(artifact_path),
                "--result",
                str(result_path),
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
                "--seed",
                str(args.seed),
                "--parity-seeds",
                str(args.parity_seeds),
                "--batch-size",
                str(args.batch_size),
                "--seq-len",
                str(args.seq_len),
                "--num-heads",
                str(args.num_heads),
                "--head-dim",
                str(args.head_dim),
            ]
            if disable_recompute:
                command.append("--disable-recompute")
            if state_v_first:
                command.append("--state-v-first")
            if external_gate:
                command.append("--external-gate")
            print(f"\nRunning {name}", flush=True)
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                results[name] = {
                    "failed": True,
                    "returncode": completed.returncode,
                }
                if not name.startswith("kda_tilelang_"):
                    raise RuntimeError(
                        f"required benchmark {name} failed with "
                        f"return code {completed.returncode}"
                    )
                print(
                    f"Optional TileLang benchmark {name} failed; continuing",
                    flush=True,
                )
                continue
            results[name] = json.loads(result_path.read_text())
            artifacts[name] = torch.load(
                artifact_path, map_location="cpu", weights_only=True
            )

        comparisons = {}
        for config_name, seed_artifacts in artifacts.items():
            operation = results[config_name]["operation"]
            state_v_first = results[config_name]["state_v_first"]
            if operation == "kda":
                reference_name = "kda_triton_recompute"
                tolerances = KDA_TOLERANCES
            elif state_v_first:
                reference_name = "gdn2_external_gate_v_first"
                tolerances = GDN2_TOLERANCES
            else:
                reference_name = "gdn2_external_gate_k_first"
                tolerances = GDN2_TOLERANCES
            references = artifacts[reference_name]
            seed_errors = {}
            for seed, artifact in seed_artifacts.items():
                field_errors = {}
                for field, tolerance in tolerances.items():
                    reference_tensor = references[seed][field]
                    candidate_tensor = artifact[field]
                    if not torch.isfinite(reference_tensor).all():
                        raise AssertionError(
                            f"triton_recompute seed {seed} {field} contains "
                            "non-finite values"
                        )
                    if not torch.isfinite(candidate_tensor).all():
                        raise AssertionError(
                            f"{config_name} seed {seed} {field} contains "
                            "non-finite values"
                        )
                    ratio = _error_ratio(reference_tensor, candidate_tensor)
                    max_abs = (
                        reference_tensor - candidate_tensor
                    ).abs().max().item()
                    field_errors[field] = {
                        "error_ratio": ratio,
                        "max_abs": max_abs,
                        "tolerance": tolerance,
                    }
                    if field == "output" and not (
                        results[config_name].get("external_gate")
                        != results[reference_name].get("external_gate")
                    ):
                        if not torch.equal(reference_tensor, candidate_tensor):
                            raise AssertionError(
                                f"{config_name} seed {seed} changed the forward "
                                f"output (max abs {max_abs:.6g})"
                            )
                    elif ratio >= tolerance and max_abs > 1e-6:
                        raise AssertionError(
                            f"{config_name} seed {seed} {field}: error ratio "
                            f"{ratio:.6g} exceeds {tolerance:.6g} "
                            f"(max abs {max_abs:.6g})"
                        )
                seed_errors[seed] = field_errors
            comparisons[config_name] = seed_errors

        # State layout is an execution choice, not a model change. Verify the
        # K-first implementation against the canonical V-first result before
        # selecting whichever is faster.
        layout_comparisons = {}
        if not args.kda_only:
            v_first_artifacts = artifacts["gdn2_external_gate_v_first"]
            k_first_artifacts = artifacts["gdn2_external_gate_k_first"]
            for seed, v_first in v_first_artifacts.items():
                seed_errors = {}
                for field, tolerance in GDN2_TOLERANCES.items():
                    k_first = k_first_artifacts[seed][field]
                    ratio = _error_ratio(v_first[field], k_first)
                    max_abs = (v_first[field] - k_first).abs().max().item()
                    seed_errors[field] = {
                        "error_ratio": ratio,
                        "max_abs": max_abs,
                        "tolerance": tolerance,
                    }
                    if ratio >= tolerance and max_abs > 1e-6:
                        raise AssertionError(
                            f"GDN2 state-layout seed {seed} {field}: error "
                            f"ratio {ratio:.6g} exceeds {tolerance:.6g} "
                            f"(max abs {max_abs:.6g})"
                        )
                layout_comparisons[seed] = seed_errors

        report = {
            "kda_only": args.kda_only,
            "shape": {
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "num_heads": args.num_heads,
                "head_dim": args.head_dim,
            },
            "parity_seeds": list(
                range(args.seed, args.seed + args.parity_seeds)
            ),
            "results": results,
            "comparisons_to_matching_recompute_reference": comparisons,
            "gdn2_k_first_to_v_first": layout_comparisons,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary_report = args.report.with_suffix(args.report.suffix + ".tmp")
        temporary_report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        temporary_report.replace(args.report)
        print("\nKDA training benchmark summary")
        print(json.dumps(report, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--op", choices=("kda", "gdn2"))
    parser.add_argument("--tilelang", choices=("0", "1"))
    parser.add_argument("--disable-recompute", action="store_true")
    parser.add_argument("--state-v-first", action="store_true")
    parser.add_argument("--external-gate", action="store_true")
    parser.add_argument(
        "--kda-only",
        action="store_true",
        help="Benchmark KDA configurations without the unrelated GDN-2 cases",
    )
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--parity-seeds", type=int, default=3)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "ablation_results/kda_training_benchmark/operator_benchmark.json"
        ),
    )
    args = parser.parse_args()
    if args.worker and (
        args.op is None
        or args.tilelang is None
        or args.artifact is None
        or args.result is None
    ):
        parser.error(
            "--worker requires --op, --tilelang, --artifact, and --result"
        )
    positive_args = {
        "batch-size": args.batch_size,
        "seq-len": args.seq_len,
        "num-heads": args.num_heads,
        "head-dim": args.head_dim,
        "warmup": args.warmup,
        "iters": args.iters,
        "parity-seeds": args.parity_seeds,
    }
    for name, value in positive_args.items():
        if value <= 0:
            parser.error(f"--{name} must be positive")
    return args


if __name__ == "__main__":
    parsed_args = _parse_args()
    if parsed_args.worker:
        _run_worker(parsed_args)
    else:
        _run_suite(parsed_args)
