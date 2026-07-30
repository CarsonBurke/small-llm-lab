"""Select the fastest validated GDN-2 state layout and run its 2K ablation.

This decision job consumes ``benchmark_kda_training.py``'s parity-gated
operator report. It runs GDN-2 only when its fastest validated state-layout
and recomputation configuration is within 20% of the fastest validated KDA
kernel; the parameter-matched GDN-2 model has a smaller MLP, so this is a
conservative whole-model speed gate.

This launches a GPU/model workload and must itself run under ``mlq``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


BENCHMARK_REPORT = Path(
    "ablation_results/delta_attention_operator_bench_reference/operator_benchmark.json"
)
DECISION_PATH = Path(
    "ablation_results/delta_attention_reference_decision/decision.json"
)
GDN2_RESULT_PATH = Path(
    "ablation_results/nanogpt_gpt2_gdn2_3to1_pm_reference_2k/result.json"
)
MAX_GDN2_TO_KDA_OPERATOR_RATIO = 1.20


def _successful_latency(results: dict, name: str) -> float | None:
    result = results.get(name)
    if not result or result.get("failed"):
        return None
    return float(result["median_ms"])


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(path)


def main() -> int:
    report = json.loads(BENCHMARK_REPORT.read_text())
    results = report["results"]

    kda_candidates = {
        name: latency
        for name in (
            "kda_triton_recompute",
            "kda_triton_no_recompute",
            "kda_tilelang_recompute",
            "kda_tilelang_no_recompute",
        )
        if (latency := _successful_latency(results, name)) is not None
    }
    gdn2_candidates = {
        name: latency
        for name in (
            "gdn2_recompute_v_first",
            "gdn2_no_recompute_v_first",
            "gdn2_recompute_k_first",
            "gdn2_no_recompute_k_first",
        )
        if (latency := _successful_latency(results, name)) is not None
    }
    if not kda_candidates or not gdn2_candidates:
        raise RuntimeError(
            "benchmark did not produce the required KDA and GDN-2 timings"
        )

    best_kda_name = min(kda_candidates, key=kda_candidates.get)
    best_gdn2_name = min(gdn2_candidates, key=gdn2_candidates.get)
    best_kda_ms = kda_candidates[best_kda_name]
    best_gdn2_ms = gdn2_candidates[best_gdn2_name]
    ratio = best_gdn2_ms / best_kda_ms
    state_v_first = best_gdn2_name.endswith("v_first")
    disable_recompute = "_no_recompute_" in best_gdn2_name

    decision = {
        "benchmark_report": str(BENCHMARK_REPORT),
        "best_kda_config": best_kda_name,
        "best_kda_median_ms": best_kda_ms,
        "best_gdn2_config": best_gdn2_name,
        "best_gdn2_median_ms": best_gdn2_ms,
        "gdn2_to_kda_operator_ratio": ratio,
        "maximum_allowed_ratio": MAX_GDN2_TO_KDA_OPERATOR_RATIO,
        "gdn2_state_v_first": state_v_first,
        "gdn2_disable_recompute": disable_recompute,
        "delta_eager_module": True,
    }
    if ratio > MAX_GDN2_TO_KDA_OPERATOR_RATIO:
        decision["action"] = "skip_gdn2_2k_too_slow"
        _atomic_write_json(DECISION_PATH, decision)
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0

    decision["action"] = "run_gdn2_2k"
    _atomic_write_json(DECISION_PATH, decision)
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)

    env = os.environ.copy()
    env.update(
        {
            "DELTA_ATTENTION_TYPE": "gdn2",
            "DELTA_DISABLE_RECOMPUTE": "1" if disable_recompute else "0",
            "DELTA_STATE_V_FIRST": "1" if state_v_first else "0",
            "DELTA_EAGER_MODULE": "1",
            "FLA_TILELANG": "0",
            "FLA_FLASH_KDA": "0",
        }
    )
    command = [
        sys.executable,
        "-u",
        "ablation.py",
        "--steps",
        "2000",
        "--name",
        "nanogpt_gpt2_gdn2_3to1_pm_reference_2k",
        "--script",
        "nanogpt_mini_gpt2vocab_kda_3to1_pm_train.py",
        "--env",
        "DELTA_ATTENTION_TYPE=gdn2",
        f"DELTA_DISABLE_RECOMPUTE={int(disable_recompute)}",
        f"DELTA_STATE_V_FIRST={int(state_v_first)}",
        "DELTA_EAGER_MODULE=1",
        "FLA_TILELANG=0",
        "FLA_FLASH_KDA=0",
    ]
    completed = subprocess.run(command, env=env, check=False)
    decision["outer_returncode"] = completed.returncode
    decision["ablation_result"] = str(GDN2_RESULT_PATH)
    if GDN2_RESULT_PATH.exists():
        ablation_result = json.loads(GDN2_RESULT_PATH.read_text())
        validation_entries = ablation_result.get("val_entries", [])
        reached_step_2000 = bool(
            validation_entries and validation_entries[-1].get("step") == 2000
        )
        training_returncode = ablation_result.get("returncode")
        succeeded = (
            completed.returncode == 0
            and training_returncode == 0
            and reached_step_2000
        )
        decision.update(
            {
                "training_returncode": training_returncode,
                "reached_step_2000": reached_step_2000,
                "final_val_bpb": ablation_result.get("final_val_bpb"),
                "outcome": "completed" if succeeded else "failed",
            }
        )
    else:
        succeeded = False
        decision.update(
            {
                "training_returncode": None,
                "reached_step_2000": False,
                "final_val_bpb": None,
                "outcome": "missing_result",
            }
        )
    _atomic_write_json(DECISION_PATH, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
