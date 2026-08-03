"""Run the 18-KDA-mixer/6-dense 2K ablation if time-to-BPB is competitive.

This script is a GPU/model workload and must run under ``mlq``. It uses the
20-step production-shape run only as an execution and rejection gate; the
architectural result still comes from the complete 2K ablation.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SPEED_RESULT = Path(
    "ablation_results/nanogpt_gpt2_kda18_dense6_mixers_speed20/"
    "result.json"
)
SHALLOW_REFERENCE = Path(
    "ablation_results/nanogpt_gpt2_kda_lowrank2070_blockcompile_2k/"
    "metrics.jsonl"
)
RUN_NAME = "nanogpt_gpt2_kda18_dense6_mixers_2k"
DECISION_PATH = Path(
    "ablation_results/kda18_dense6_mixers_decision/decision.json"
)
MAXIMUM_STEP_MS = 3200.0
REQUIRED_TIME_BPB_IMPROVEMENT = 0.005


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    temporary_path.replace(path)


def interpolate_reference_bpb(train_time_ms: float) -> float:
    validation_entries = [
        json.loads(line)
        for line in SHALLOW_REFERENCE.read_text().splitlines()
        if json.loads(line).get("type") == "val"
    ]
    if not validation_entries:
        raise RuntimeError("Shallow KDA reference has no validation entries")
    if train_time_ms <= validation_entries[0]["train_time_ms"]:
        return float(validation_entries[0]["val_bpb"])
    for left, right in zip(
        validation_entries,
        validation_entries[1:],
        strict=False,
    ):
        left_time = float(left["train_time_ms"])
        right_time = float(right["train_time_ms"])
        if train_time_ms <= right_time:
            fraction = (train_time_ms - left_time) / (
                right_time - left_time
            )
            return float(left["val_bpb"]) + fraction * (
                float(right["val_bpb"]) - float(left["val_bpb"])
            )
    return float(validation_entries[-1]["val_bpb"])


def main() -> int:
    speed_result = json.loads(SPEED_RESULT.read_text())
    validations = speed_result.get("val_entries", [])
    final_validation = validations[-1] if validations else {}
    step_ms = float(final_validation.get("step_avg_ms", float("inf")))
    train_time_ms = float(
        final_validation.get("train_time_ms", float("inf"))
    )
    candidate_bpb = float(final_validation.get("val_bpb", float("inf")))
    reference_bpb = interpolate_reference_bpb(train_time_ms)
    time_bpb_improvement = reference_bpb - candidate_bpb

    decision = {
        "speed_result": str(SPEED_RESULT),
        "shallow_reference": str(SHALLOW_REFERENCE),
        "measured_step_ms": step_ms,
        "maximum_step_ms": MAXIMUM_STEP_MS,
        "measured_step20_bpb": candidate_bpb,
        "reference_bpb_at_same_training_time": reference_bpb,
        "time_bpb_improvement": time_bpb_improvement,
        "required_time_bpb_improvement": REQUIRED_TIME_BPB_IMPROVEMENT,
    }
    if (
        speed_result.get("returncode") != 0
        or step_ms > MAXIMUM_STEP_MS
        or time_bpb_improvement < REQUIRED_TIME_BPB_IMPROVEMENT
    ):
        decision["action"] = "skip_2k"
        atomic_write_json(DECISION_PATH, decision)
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0

    decision["action"] = "run_2k"
    atomic_write_json(DECISION_PATH, decision)
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
    command = [
        sys.executable,
        "-u",
        "scripts/ablation.py",
        "--steps",
        "2000",
        "--val-every",
        "20",
        "--name",
        RUN_NAME,
        "--script",
        "pretraining/nanogpt_mini/nanogpt_mini_gpt2vocab_kda_3to1_pm_train.py",
        "--env",
        "NUM_LAYERS=24",
        "--env",
        "DELTA_ATTENTION_TYPE=kda",
        "--env",
        "DELTA_MLP_ON_DELTA=0",
        "--env",
        "MLP_HIDDEN=2048",
        "--env",
        "DELTA_DISABLE_RECOMPUTE=1",
        "--env",
        "DELTA_STATE_V_FIRST=1",
        "--env",
        "DELTA_EAGER_MODULE=1",
        "--env",
        "DELTA_BLOCKWISE_COMPILE=1",
        "--env",
        "DELTA_COMPILE_MODE=default",
        "--env",
        "KDA_COMPILE_DIAGNOSTICS=0",
        "--env",
        "FLA_TILELANG=0",
        "--env",
        "FLA_FLASH_KDA=0",
        "--env",
        "MBS=8",
    ]
    completed = subprocess.run(command, check=False)
    result_path = Path("ablation_results") / RUN_NAME / "result.json"
    decision["outer_returncode"] = completed.returncode
    decision["result_path"] = str(result_path)
    if result_path.exists():
        result = json.loads(result_path.read_text())
        result_validations = result.get("val_entries", [])
        decision["final_val_bpb"] = result.get("final_val_bpb")
        decision["training_returncode"] = result.get("returncode")
        decision["reached_step_2000"] = bool(
            result_validations
            and result_validations[-1].get("step") == 2000
        )
    else:
        decision["final_val_bpb"] = None
        decision["training_returncode"] = None
        decision["reached_step_2000"] = False
    succeeded = (
        completed.returncode == 0
        and decision["training_returncode"] == 0
        and decision["reached_step_2000"]
    )
    decision["outcome"] = "completed" if succeeded else "failed"
    atomic_write_json(DECISION_PATH, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
