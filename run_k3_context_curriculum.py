"""Run the measured KDA winner through a 2K -> 4K -> 8K curriculum.

This script launches three foreground training stages and must itself run
inside one mlq job. The total global batch remains 524,288 tokens and every
stage keeps 32,768 tokens per microbatch:

* steps 0..6000: context 2048, MBS 16
* steps 6000..7500: context 4096, MBS 8
* steps 7500..8000: context 8192, MBS 4

Each boundary carries model, optimizer, RNG, and exact dataset position.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


STAGES = (
    ("ctx2k", 6000, 2048, 16),
    ("ctx4k", 7500, 4096, 8),
    ("ctx8k", 8000, 8192, 4),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/datasets/k3mix_v5_gpt2_8k")
    parser.add_argument(
        "--cooldown-data",
        help="optional high-quality mix used only for the final 8K-context stage",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--script",
        default="nanogpt_mini_gpt2vocab_kda_3to1_pm_train.py",
    )
    parser.add_argument("--kda-heads", type=int, default=4)
    parser.add_argument("--full-rank-gate", action="store_true")
    parser.add_argument("--per-head-muon", action="store_true")
    parser.add_argument("--mtp-heads", type=int, default=0)
    parser.add_argument("--mtp-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--dense-attention",
        choices=("mha", "gated_nope_mla"),
        default="mha",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=("stable_linear", "cosine"),
        default="stable_linear",
    )
    parser.add_argument("--embed-lr", type=float, default=0.7)
    parser.add_argument("--proj-lr", type=float, default=0.004)
    parser.add_argument("--kda-conv-lr", type=float, default=0.004)
    parser.add_argument("--scalar-lr", type=float, default=0.015)
    parser.add_argument("--muon-lr", type=float, default=0.025)
    parser.add_argument("--adam-weight-decay", type=float, default=0.001)
    parser.add_argument("--muon-weight-decay", type=float, default=0.05)
    args = parser.parse_args()
    if args.mtp_heads < 0:
        parser.error("--mtp-heads must be nonnegative")
    if args.mtp_loss_weight < 0:
        parser.error("--mtp-loss-weight must be nonnegative")
    if min(
        args.embed_lr,
        args.proj_lr,
        args.kda_conv_lr,
        args.scalar_lr,
        args.muon_lr,
    ) < 0:
        parser.error("learning rates must be nonnegative")

    manifest_path = Path(args.data) / "mix_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["training_steps"] < STAGES[-1][1]:
        raise ValueError(
            f"{manifest_path} contains {manifest['training_steps']} steps, "
            f"but the curriculum requires {STAGES[-1][1]}"
        )
    cooldown_data = Path(args.cooldown_data) if args.cooldown_data else None
    if cooldown_data is not None:
        cooldown_manifest_path = cooldown_data / "mix_manifest.json"
        cooldown_manifest = json.loads(cooldown_manifest_path.read_text())
        cooldown_steps = int(cooldown_manifest["training_steps"])
        final_stage_steps = STAGES[-1][1] - STAGES[-2][1]
        if cooldown_steps < final_stage_steps:
            raise ValueError(
                f"{cooldown_manifest_path} contains {cooldown_steps} steps, "
                f"but the final stage requires {final_stage_steps}"
            )

    resume_checkpoint = None
    for stage_name, stop_after, seq_len, mbs in STAGES:
        stage_run_id = f"{args.run_id}_{stage_name}"
        stage_data = (
            cooldown_data
            if cooldown_data is not None and stage_name == STAGES[-1][0]
            else Path(args.data)
        )
        env = os.environ.copy()
        # Stage 1 must always start from scratch. Do not let an unrelated
        # shell-level resume setting silently change the campaign.
        env.pop("RESUME_CHECKPOINT", None)
        env.update(
            {
                "RUN_ID": stage_run_id,
                "DATA_PATH": str(stage_data),
                "ITERATIONS": str(STAGES[-1][1]),
                "STOP_AFTER_STEP": str(stop_after),
                "SAVE_RESUME_STATE": "1",
                "SEQ_LEN": str(seq_len),
                "MBS": str(mbs),
                "NUM_LAYERS": "8",
                "DELTA_LAYER_INDICES": "0,1,2,4,5,6",
                "DELTA_ATTENTION_TYPE": "kda",
                "DELTA_MLP_ON_DELTA": "0",
                "MLP_HIDDEN": "2070",
                "KDA_NUM_HEADS": str(args.kda_heads),
                "KDA_FULL_RANK_GATE": "1" if args.full_rank_gate else "0",
                "PER_HEAD_MUON": "1" if args.per_head_muon else "0",
                "MTP_NUM_HEADS": str(args.mtp_heads),
                "MTP_LOSS_WEIGHT": str(args.mtp_loss_weight),
                "DENSE_ATTENTION_TYPE": args.dense_attention,
                "LR_SCHEDULE": args.lr_schedule,
                "EMBED_LR": str(args.embed_lr),
                "PROJ_LR": str(args.proj_lr),
                "KDA_CONV_LR": str(args.kda_conv_lr),
                "SCALAR_LR": str(args.scalar_lr),
                "MUON_LR": str(args.muon_lr),
                "ADAM_WEIGHT_DECAY": str(args.adam_weight_decay),
                "MUON_WEIGHT_DECAY": str(args.muon_weight_decay),
                "DELTA_DISABLE_RECOMPUTE": "1",
                "DELTA_STATE_V_FIRST": "1",
                "DELTA_EAGER_MODULE": "1",
                "DELTA_BLOCKWISE_COMPILE": "0",
                "DELTA_COMPILE_MODE": "default",
                "DOMAIN_VAL_EVERY": "200",
            }
        )
        if resume_checkpoint is not None:
            env["RESUME_CHECKPOINT"] = str(resume_checkpoint)
        subprocess.run(
            [sys.executable, "-u", args.script],
            env=env,
            check=True,
        )
        resume_checkpoint = Path("logs") / f"{stage_run_id}_resume.pt"
        if not resume_checkpoint.exists():
            raise FileNotFoundError(
                f"stage {stage_name} did not write {resume_checkpoint}"
            )

    print(f"final model: logs/{args.run_id}_ctx8k_final_model.pt")


if __name__ == "__main__":
    main()
