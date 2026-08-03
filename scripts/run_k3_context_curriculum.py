"""Run the measured KDA base through a 2K -> 4K -> 8K curriculum.

This script launches three foreground training stages and must itself run
inside one mlq job. The total global batch remains 524,288 tokens and every
stage keeps 32,768 tokens per microbatch. By default, 75% of updates use 2K
context, 18.75% use 4K, and the final 6.25% use 8K.

Each boundary carries model, optimizer, RNG, and exact dataset position.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ablation import MetricsWriter, parse_log_line

REPO_ROOT = Path(__file__).resolve().parents[1]


def relay_stage_output(
    process: subprocess.Popen,
    metrics: MetricsWriter,
    time_offset_ms: float,
) -> float:
    """Relay one child stage and return its measured training-time span."""
    stage_time_ms = 0.0
    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            entry = parse_log_line(line.strip())
            if entry is not None:
                if "train_time_ms" in entry:
                    raw_time_ms = float(entry["train_time_ms"])
                    stage_time_ms = max(stage_time_ms, raw_time_ms)
                    entry = {
                        **entry,
                        "train_time_ms": raw_time_ms + time_offset_ms,
                    }
                metrics.write_entry(entry)
        returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, process.args)
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    return stage_time_ms


def curriculum_stages(total_steps: int) -> tuple[tuple[str, int, int, int], ...]:
    ctx2k_stop = round(total_steps * 0.75)
    ctx4k_stop = round(total_steps * 0.9375)
    if not 0 < ctx2k_stop < ctx4k_stop < total_steps:
        raise ValueError(
            f"{total_steps} steps is too short for the three-stage curriculum"
        )
    return (
        ("ctx2k", ctx2k_stop, 2048, 16),
        ("ctx4k", ctx4k_stop, 4096, 8),
        ("ctx8k", total_steps, 8192, 4),
    )


def validate_training_manifest(
    manifest: dict,
    total_steps: int,
    expected_batch_tokens: int = 524_288,
) -> None:
    if manifest.get("train_batch_tokens") != expected_batch_tokens:
        raise ValueError(
            "dataset batch geometry does not match the trainer: "
            f"{manifest.get('train_batch_tokens')} != {expected_batch_tokens}"
        )
    if manifest.get("loader_aligned") is not True:
        raise ValueError("dataset is not marked loader-aligned")
    required_tokens = total_steps * expected_batch_tokens + 1
    if manifest.get("unique_stream_tokens", 0) < required_tokens:
        raise ValueError(
            "dataset is too short for one-pass training: "
            f"{manifest.get('unique_stream_tokens', 0)} < {required_tokens} tokens"
        )
    available_steps = sum(
        int(shard.get("steps", 0)) for shard in manifest.get("shards", ())
    )
    if available_steps < total_steps:
        raise ValueError(
            "dataset has too few loader-aligned steps for one-pass training: "
            f"{available_steps} < {total_steps}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/datasets/k3mix_v5_gpt2_8k")
    parser.add_argument(
        "--cooldown-data",
        help="optional high-quality mix used only for the final 8K-context stage",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument(
        "--script",
        default="pretraining/nanogpt_mini/nanogpt_mini_gpt2vocab_kda_3to1_pm_train.py",
    )
    parser.add_argument("--kda-heads", type=int, default=3)
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
        default="cosine",
    )
    parser.add_argument("--warmup-fraction", type=float, default=0.01)
    parser.add_argument("--embed-lr", type=float, default=0.7)
    parser.add_argument("--proj-lr", type=float, default=0.004)
    parser.add_argument("--kda-conv-lr", type=float, default=0.004)
    parser.add_argument("--scalar-lr", type=float, default=0.015)
    parser.add_argument("--muon-lr", type=float, default=0.025)
    parser.add_argument("--adam-weight-decay", type=float, default=0.001)
    parser.add_argument("--muon-weight-decay", type=float, default=0.05)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-momentum-warmup-start", type=float, default=0.85)
    parser.add_argument("--muon-momentum-warmup-steps", type=int, default=500)
    parser.add_argument("--val-loss-every", type=int, default=250)
    parser.add_argument("--domain-val-every", type=int, default=1000)
    parser.add_argument("--val-tokens", type=int, default=4 * 524_288)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.kda_heads <= 0:
        parser.error("--kda-heads must be positive")
    stages = curriculum_stages(args.steps)
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
    if not 0 <= args.warmup_fraction < 1:
        parser.error("--warmup-fraction must be in [0, 1)")
    if min(args.adam_weight_decay, args.muon_weight_decay) < 0:
        parser.error("weight decays must be nonnegative")
    if not (
        0
        <= args.muon_momentum_warmup_start
        <= args.muon_momentum
        < 1
    ):
        parser.error(
            "Muon momentum must satisfy "
            "0 <= warmup start <= momentum < 1"
        )
    if args.muon_momentum_warmup_steps < 0:
        parser.error("--muon-momentum-warmup-steps must be nonnegative")
    if min(args.val_loss_every, args.domain_val_every, args.val_tokens) <= 0:
        parser.error("validation intervals and token count must be positive")
    if args.domain_val_every % args.val_loss_every:
        parser.error(
            "--domain-val-every must be a multiple of --val-loss-every"
        )
    microbatch_tokens = stages[0][2] * stages[0][3]
    if args.val_tokens % microbatch_tokens:
        parser.error(
            f"--val-tokens must be divisible by {microbatch_tokens}"
        )

    manifest_path = Path(args.data) / "mix_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_training_manifest(manifest, args.steps)
    cooldown_data = Path(args.cooldown_data) if args.cooldown_data else None
    if cooldown_data is not None:
        cooldown_manifest_path = cooldown_data / "mix_manifest.json"
        cooldown_manifest = json.loads(cooldown_manifest_path.read_text())
        cooldown_steps = int(cooldown_manifest["training_steps"])
        final_stage_steps = stages[-1][1] - stages[-2][1]
        if cooldown_steps < final_stage_steps:
            raise ValueError(
                f"{cooldown_manifest_path} contains {cooldown_steps} steps, "
                f"but the final stage requires {final_stage_steps}"
            )

    metrics_dir = Path("ablation_results") / args.run_id
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError(
            f"{metrics_path} already exists; choose a fresh --run-id"
        )
    metrics = MetricsWriter(metrics_path, args.run_id)
    resume_checkpoint = None
    training_time_offset_ms = 0.0
    try:
        for stage_name, stop_after, seq_len, mbs in stages:
            stage_run_id = f"{args.run_id}_{stage_name}"
            stage_data = (
                cooldown_data
                if cooldown_data is not None and stage_name == stages[-1][0]
                else Path(args.data)
            )
            env = os.environ.copy()
            python_path = env.get("PYTHONPATH")
            env["PYTHONPATH"] = (
                f"{REPO_ROOT}{os.pathsep}{python_path}"
                if python_path
                else str(REPO_ROOT)
            )
            # Stage 1 must always start from scratch. Do not let an unrelated
            # shell-level resume setting silently change the campaign.
            env.pop("RESUME_CHECKPOINT", None)
            env.update(
                {
                    "RUN_ID": stage_run_id,
                    "DATA_PATH": str(stage_data),
                    "ITERATIONS": str(args.steps),
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
                    "WARMUP_FRACTION": str(args.warmup_fraction),
                    "EMBED_LR": str(args.embed_lr),
                    "PROJ_LR": str(args.proj_lr),
                    "KDA_CONV_LR": str(args.kda_conv_lr),
                    "SCALAR_LR": str(args.scalar_lr),
                    "MUON_LR": str(args.muon_lr),
                    "ADAM_WEIGHT_DECAY": str(args.adam_weight_decay),
                    "MUON_WEIGHT_DECAY": str(args.muon_weight_decay),
                    "MUON_MOMENTUM": str(args.muon_momentum),
                    "MUON_MOMENTUM_WARMUP_START": str(
                        args.muon_momentum_warmup_start
                    ),
                    "MUON_MOMENTUM_WARMUP_STEPS": str(
                        args.muon_momentum_warmup_steps
                    ),
                    "DELTA_DISABLE_RECOMPUTE": "1",
                    "DELTA_STATE_V_FIRST": "1",
                    "DELTA_EAGER_MODULE": "1",
                    "DELTA_BLOCKWISE_COMPILE": "0",
                    "DELTA_COMPILE_MODE": "default",
                    "VAL_LOSS_EVERY": str(args.val_loss_every),
                    "DOMAIN_VAL_EVERY": str(args.domain_val_every),
                    "VAL_TOKENS": str(args.val_tokens),
                }
            )
            if resume_checkpoint is not None:
                env["RESUME_CHECKPOINT"] = str(resume_checkpoint)
                # The prior stage already emitted validation at this exact
                # global step. Avoid paying for and recording it twice.
                env["SKIP_INITIAL_VALIDATION"] = "1"
            process = subprocess.Popen(
                [sys.executable, "-u", args.script],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            stage_time_ms = relay_stage_output(
                process,
                metrics,
                training_time_offset_ms,
            )
            training_time_offset_ms += stage_time_ms
            resume_checkpoint = Path("logs") / f"{stage_run_id}_resume.pt"
            if not resume_checkpoint.exists():
                raise FileNotFoundError(
                    f"stage {stage_name} did not write {resume_checkpoint}"
                )
    finally:
        metrics.close()

    print(f"final model: logs/{args.run_id}_ctx8k_final_model.pt")


if __name__ == "__main__":
    main()
