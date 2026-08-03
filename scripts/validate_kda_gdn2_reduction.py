"""Validate that the released GDN-2 kernel recovers KDA for tied gates.

This is a production-shape CUDA correctness workload and must run through
``mlq``. It compares outputs and every differentiable input gradient under
the bounded KDA decay used by the KDA-centered GDN-2 ablation.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

# Match the training jobs. FLA resolves and caches backend dispatch at import.
os.environ["FLA_DISABLE_BACKEND_DISPATCH"] = "0"
os.environ["FLA_FLASH_KDA"] = "0"
os.environ["FLA_TILELANG"] = "0"

import torch
from fla.ops.gdn2 import chunk_gdn2
from fla.ops.kda import chunk_kda
from fla.ops.kda.backends import kda_registry


SHAPE = (8, 1024, 3, 128)
SEED = 1337
STATE_V_FIRST = os.environ.get("DELTA_STATE_V_FIRST", "1") == "1"
OUTPUT_TOLERANCE = 0.005
GRADIENT_TOLERANCE = 0.025
RESULT_PATH = Path(
    "ablation_results/kda_gdn2_reduction_parity/result.json"
)


def make_leaves() -> dict[str, torch.Tensor]:
    torch.manual_seed(SEED)
    leaves = {
        name: torch.randn(
            SHAPE,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        for name in ("q", "k", "v", "g")
    }
    leaves["beta"] = torch.randn(
        SHAPE[:-1],
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    leaves["A_log"] = torch.zeros(
        SHAPE[2],
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    log_dt = torch.empty(
        SHAPE[2] * SHAPE[3],
        device="cuda",
        dtype=torch.float32,
    ).uniform_(math.log(0.001), math.log(0.1))
    dt = log_dt.exp().clamp(min=1e-4)
    leaves["dt_bias"] = (
        dt + torch.log(-torch.expm1(-dt))
    ).requires_grad_()
    return leaves


def clone_leaves(
    leaves: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone().requires_grad_()
        for name, tensor in leaves.items()
    }


def run_kda(
    leaves: dict[str, torch.Tensor],
    output_grad: torch.Tensor,
) -> dict[str, torch.Tensor]:
    output, _ = chunk_kda(
        q=leaves["q"],
        k=leaves["k"],
        v=leaves["v"],
        g=leaves["g"],
        beta=leaves["beta"],
        A_log=leaves["A_log"],
        dt_bias=leaves["dt_bias"],
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        state_v_first=STATE_V_FIRST,
        disable_recompute=True,
    )
    output.backward(output_grad)
    return {
        "output": output.detach().float().cpu(),
        **{
            f"{name}_grad": tensor.grad.detach().float().cpu()
            for name, tensor in leaves.items()
        },
    }


def run_gdn2(
    leaves: dict[str, torch.Tensor],
    output_grad: torch.Tensor,
) -> dict[str, torch.Tensor]:
    beta = (
        leaves["beta"]
        .sigmoid()
        .to(leaves["v"].dtype)
        .unsqueeze(-1)
        .expand(SHAPE)
    )
    output, _ = chunk_gdn2(
        q=leaves["q"],
        k=leaves["k"],
        v=leaves["v"],
        g=leaves["g"],
        b=beta,
        w=beta,
        A_log=leaves["A_log"],
        dt_bias=leaves["dt_bias"],
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        state_v_first=STATE_V_FIRST,
        disable_recompute=True,
    )
    output.backward(output_grad)
    return {
        "output": output.detach().float().cpu(),
        **{
            f"{name}_grad": tensor.grad.detach().float().cpu()
            for name, tensor in leaves.items()
        },
    }


def relative_rmse(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> float:
    error = (reference - candidate).flatten().square().mean().sqrt()
    scale = reference.flatten().square().mean().sqrt()
    return float(error / (scale + 1e-8))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("KDA/GDN-2 reduction validation requires CUDA")

    base_leaves = make_leaves()
    torch.manual_seed(SEED + 1)
    output_grad = torch.randn(
        SHAPE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    kda = run_kda(clone_leaves(base_leaves), output_grad)
    gdn2 = run_gdn2(clone_leaves(base_leaves), output_grad)
    torch.cuda.synchronize()

    comparisons = {}
    for field, reference in kda.items():
        candidate = gdn2[field]
        if not torch.isfinite(reference).all():
            raise AssertionError(f"KDA {field} is non-finite")
        if not torch.isfinite(candidate).all():
            raise AssertionError(f"GDN-2 {field} is non-finite")
        ratio = relative_rmse(reference, candidate)
        max_abs = float((reference - candidate).abs().max())
        tolerance = (
            OUTPUT_TOLERANCE
            if field == "output"
            else GRADIENT_TOLERANCE
        )
        comparisons[field] = {
            "relative_rmse": ratio,
            "max_abs": max_abs,
            "tolerance": tolerance,
        }
        if ratio >= tolerance and max_abs > 1e-6:
            raise AssertionError(
                f"{field}: relative RMSE {ratio:.6g} exceeds "
                f"{tolerance:.6g}; max abs {max_abs:.6g}"
            )

    result = {
        "shape": list(SHAPE),
        "seed": SEED,
        "state_v_first": STATE_V_FIRST,
        "passed": True,
        "kda_dispatch_records": sorted(kda_registry._logged),
        "comparisons": comparisons,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULT_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    temporary_path.replace(RESULT_PATH)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
