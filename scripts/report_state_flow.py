"""Read-only matched-budget report for completed shared-state ablations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(run):
    directory = ROOT / "ablation_results" / run
    config = json.loads((directory / "config.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    if any(result[k] != value for k, value in
           (("status", "completed"), ("completed_steps", 1000), ("final_val_step", 1000))):
        raise ValueError(f"{run} lacks a completed 1000-update result")
    records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines() if line]
    vals = [r for r in records if r.get("type") == "val"]
    if [v["step"] for v in vals] != list(range(0, 1001, 20)):
        raise ValueError(f"{run} validation panel is incomplete or duplicated")
    if not all(math.isfinite(v[k]) for v in vals for k in ("val_bpb", "val_bpb_pass1")):
        raise ValueError(f"{run} has nonfinite validation")
    if vals[-1]["step"] != 1000 or vals[-1]["val_bpb"] != result["final_val_bpb"]:
        raise ValueError(f"{run} canonical metrics disagree")
    timing = Path(config["throughput_gate"]["path"])
    if hashlib.sha256(timing.read_bytes()).hexdigest() != config["throughput_gate"]["sha256"]:
        raise ValueError(f"{run} throughput evidence changed")
    evidence = json.loads(timing.read_text())
    return config, result, vals, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="gdn2_dense_flow128_1k")
    parser.add_argument("--control", default="gdn2_layer_top2_128_1k")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate, control = load(args.candidate), load(args.control)
    fields = ("steps", "val_every", "seed", "batch_tokens", "microbatch", "seq_len", "segment_size",
              "data_path", "tokenizer", "tokenizer_sha256", "val_tokens", "val_bytes", "optimizer",
              "precision", "bptt", "loss_reduction", "autotuning_policy", "torch", "gpu")
    matched = {key: candidate[0][key] == control[0][key] for key in fields}
    allowed = {"pool_reads", "pool_readout", "pool_balance_coefficient"}
    cmodel, rmodel = candidate[0]["model_config"], control[0]["model_config"]
    matched["remaining_model_config"] = ({k: v for k, v in cmodel.items() if k not in allowed}
                                         == {k: v for k, v in rmodel.items() if k not in allowed})
    matched["implementation_sources"] = candidate[3]["source_sha256"] == control[3]["source_sha256"]
    if not all(matched.values()):
        raise ValueError(f"unmatched comparison: {[k for k, v in matched.items() if not v]}")
    improvement = control[1]["final_val_bpb"] - candidate[1]["final_val_bpb"]
    arms = []
    for name, (config, result, vals, evidence) in ((args.candidate, candidate), (args.control, control)):
        arms.append(dict(run=name, final_bpb=result["final_val_bpb"],
                         pass_one_bpb=vals[-1]["val_bpb_pass1"],
                         shared_read_gain_bpb=vals[-1]["val_bpb_pass1"] - vals[-1]["val_bpb"],
                         parameters=config["parameters"],
                         median_update_seconds=evidence["candidate"]["median_update_seconds"],
                         peak_reserved_mib=evidence["candidate"]["peak_reserved_mib"],
                         plain_mini_speed_gate_passed=evidence["gate_passed"]))
    report = dict(matched_fields=matched, arms=arms, improvement_bpb=improvement,
                  quality_gate_passed=improvement > .005,
                  threshold_bpb=.005, replicates_per_arm=1,
                  comparison_scope="all-bank access, concatenated readout and zero balance coefficient versus top-two summed reads with balance regularization",
                  submission_ready=False,
                  limitations=["one seed, no confidence interval", "readout capacity, access and balance regularization change together",
                               "plain-mini speed gate not passed", "quantized 16 MB and 8xH100 limits unvalidated"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        for label, arm, color in (("All six banks + concat", candidate, "#116d96"),
                                  ("Top two banks + sum", control, "#bd6419")):
            vals = [v for v in arm[2] if v["step"] >= 400]
            axes[0].plot([v["step"] for v in vals], [v["val_bpb"] for v in vals], label=label, color=color)
            axes[1].plot([v["step"] for v in vals],
                         [v["val_bpb_pass1"] - v["val_bpb"] for v in vals], label=label, color=color)
        axes[0].set(title="Matched 1000-update validation", xlabel="Optimizer update", ylabel="Bits per byte (lower is better)")
        axes[1].set(title="Checkpoint benefit of shared reads", xlabel="Optimizer update", ylabel="Pass-one BPB minus full BPB")
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend(frameon=False)
        fig.suptitle(f"Dense state flow: {improvement:+.5f} BPB improvement over control", fontweight="bold")
        fig.savefig(args.output / "learning_curves.png", dpi=180)
        plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
